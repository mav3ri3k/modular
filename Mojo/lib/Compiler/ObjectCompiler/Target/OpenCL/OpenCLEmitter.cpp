// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
#include "OpenCLEmitter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/ScalarizeMaskedMemIntrin.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cctype>

using namespace llvm;
namespace M::KGEN {
namespace {
// Preserve the logical argument boundary in the manifest while exposing typed
// leaves to OpenCL. Generic LLVM inlining/SROA removes reconstruction storage;
// no host register coercion or pointer-to-integer conversion is involved.
using AggregateABIs =
    DenseMap<const Function *, std::vector<M::OpenCL::ArgumentABI>>;
// Work-item branches can diverge. Keep mask predicates as lane values when
// LLVM legalizes masked memory, rather than using CPU-specific packed masks.
class OpenCLLegalizationTTI
    : public TargetTransformInfoImplCRTPBase<OpenCLLegalizationTTI> {
public:
  explicit OpenCLLegalizationTTI(const DataLayout &dl)
      : TargetTransformInfoImplCRTPBase(dl) {}
  bool hasBranchDivergence(const Function * = nullptr) const override {
    return true;
  }
};

class LegalizePointerVectorsPass
#if LLVM_VERSION_MAJOR >= 23
    : public RequiredPassInfoMixin<LegalizePointerVectorsPass> {
#else
    : public PassInfoMixin<LegalizePointerVectorsPass> {
#endif
public:
  PreservedAnalyses run(Function &f, FunctionAnalysisManager &) {
    SmallVector<Instruction *> work;
    for (auto &i : instructions(f))
      work.push_back(&i);
    for (auto *i : work) {
      IRBuilder<> builder(i);
      if (auto *call = dyn_cast<IntrinsicInst>(i);
          call && call->getIntrinsicID() == Intrinsic::stepvector) {
        if (auto *ty = dyn_cast<FixedVectorType>(call->getType())) {
          SmallVector<Constant *> lanes;
          for (unsigned n = 0; n < ty->getNumElements(); ++n)
            lanes.push_back(ConstantInt::get(ty->getElementType(), n));
          call->replaceAllUsesWith(ConstantVector::get(lanes));
          call->eraseFromParent();
          continue;
        }
      }
      auto *ty = dyn_cast<FixedVectorType>(i->getType());
      if (ty && ty->getElementType()->isPointerTy() &&
          (isa<GetElementPtrInst, IntToPtrInst, SelectInst, FreezeInst>(i))) {
        auto lane = [&](Value *v, unsigned n) -> Value * {
          return v->getType()->isVectorTy() ? builder.CreateExtractElement(v, n)
                                            : v;
        };
        Value *result = PoisonValue::get(ty);
        for (unsigned n = 0; n < ty->getNumElements(); ++n) {
          Value *scalar;
          if (auto *gep = dyn_cast<GetElementPtrInst>(i)) {
            SmallVector<Value *> indices;
            for (auto &index : gep->indices())
              indices.push_back(lane(index.get(), n));
            scalar =
                builder.CreateGEP(gep->getSourceElementType(),
                                  lane(gep->getPointerOperand(), n), indices);
          } else if (auto *cast = dyn_cast<IntToPtrInst>(i)) {
            scalar = builder.CreateIntToPtr(lane(cast->getOperand(0), n),
                                            ty->getElementType());
          } else if (auto *select = dyn_cast<SelectInst>(i)) {
            scalar = builder.CreateSelect(lane(select->getCondition(), n),
                                          lane(select->getTrueValue(), n),
                                          lane(select->getFalseValue(), n));
          } else {
            scalar = builder.CreateFreeze(lane(i->getOperand(0), n));
          }
          result = builder.CreateInsertElement(result, scalar, n);
        }
        i->replaceAllUsesWith(result);
        i->eraseFromParent();
        continue;
      }
      // Recover typed address provenance after lane extraction has simplified
      // vectorized base-plus-byte-offset arithmetic. Arbitrary casts still
      // fail.
      auto *cast = dyn_cast<IntToPtrInst>(i);
      if (!cast || !cast->getType()->isPointerTy())
        continue;
      Value *address = cast->getOperand(0);
      ConstantInt *index = nullptr;
      if (auto *extract = dyn_cast<ExtractElementInst>(address)) {
        index = dyn_cast<ConstantInt>(extract->getIndexOperand());
        if (!index)
          continue;
        address = extract->getVectorOperand();
      }
      auto *sum = dyn_cast<BinaryOperator>(address);
      if (!sum || sum->getOpcode() != Instruction::Add)
        continue;
      for (unsigned n = 0; n < 2; ++n) {
        Value *candidate = sum->getOperand(n);
        if (index)
          candidate = findScalarElement(candidate, index->getZExtValue());
        auto *base = dyn_cast_or_null<PtrToIntInst>(candidate);
        if (!base || base->getType()->getIntegerBitWidth() != 64 ||
            base->getPointerOperand()->getType() != cast->getType())
          continue;
        Value *offset = sum->getOperand(1 - n);
        if (index)
          offset = builder.CreateExtractElement(offset, index);
        auto *pointer = builder.CreateGEP(builder.getInt8Ty(),
                                          base->getPointerOperand(), offset);
        cast->replaceAllUsesWith(pointer);
        cast->eraseFromParent();
        break;
      }
    }
    return PreservedAnalyses::none();
  }
};

// Expose aggregate freeze leaves so InstCombine can remove reconstruction.
// This preserves each field's stable choice without requiring aggregate C ABI.
static Value *freezeLeaves(IRBuilder<> &builder, Value *value, Type *ty) {
  unsigned count = 0;
  if (auto *record = dyn_cast<StructType>(ty))
    count = record->getNumElements();
  else if (auto *array = dyn_cast<ArrayType>(ty))
    count = array->getNumElements();
  else
    return builder.CreateFreeze(value);
  Value *result = PoisonValue::get(ty);
  for (unsigned i = 0; i < count; ++i) {
    auto *field = builder.CreateExtractValue(value, i);
    result = builder.CreateInsertValue(
        result, freezeLeaves(builder, field, field->getType()), i);
  }
  return result;
}
static Error flattenKernelArguments(Module &module, AggregateABIs &abis) {
  using ABI = M::OpenCL::ArgumentABI;
  SmallVector<Function *> kernels;
  for (auto &f : module)
    if (!f.isDeclaration() && (f.getCallingConv() == CallingConv::SPIR_KERNEL ||
                               f.hasFnAttribute("mojo.opencl.kernel")))
      kernels.push_back(&f);
  auto &dl = module.getDataLayout();
  for (auto *f : kernels) {
    bool aggregate = false;
    for (auto &a : f->args())
      aggregate |= a.getType()->isAggregateType() || a.hasByValAttr();

    if (!f->use_empty())
      return createStringError(
          inconvertibleErrorCode(),
          "OpenCL: aggregate kernels cannot be called directly");
    std::vector<ABI> roots;
    SmallVector<Type *> types;
    SmallVector<Type *> originals;
    bool invalid = false;
    for (auto &a : f->args()) {
      Type *t = a.hasByValAttr() ? a.getParamByValType() : a.getType();
      if (!t->isSized() || dl.getTypeAllocSize(t).isScalable())
        return createStringError(inconvertibleErrorCode(),
                                 "OpenCL: unsized/scalable kernel argument");
      originals.push_back(t);
      ABI root{ABI::Aggregate, 0};
      uint64_t bytes = dl.getTypeAllocSize(t).getFixedValue();
      if (bytes > 65536 || !bytes)
        invalid = true;
      root.size = root.kind == ABI::Aggregate ? uint32_t(alignTo(bytes, 8) * 4)
                  : root.kind == ABI::BufferView ? 24
                                                 : uint32_t(bytes);
      auto flatten = [&](auto &self, Type *type, uint64_t offset) -> void {
        if (auto *st = dyn_cast<StructType>(type)) {
          if (st->isOpaque()) {
            invalid = true;
            return;
          }
          auto *layout = dl.getStructLayout(st);
          for (unsigned i = 0; i < st->getNumElements(); ++i)
            self(self, st->getElementType(i),
                 offset + layout->getElementOffset(i));
        } else if (auto *array = dyn_cast<ArrayType>(type)) {
          auto stride =
              dl.getTypeAllocSize(array->getElementType()).getFixedValue();
          if (array->getNumElements() > 256) {
            invalid = true;
            return;
          }
          for (unsigned i = 0; i < array->getNumElements(); ++i)
            self(self, array->getElementType(), offset + stride * i);
        } else {
          if (types.size() >= 256) {
            invalid = true;
            return;
          }
          types.push_back(type);
          if (root.kind == ABI::Aggregate)
            root.fields.push_back(
                {uint32_t(offset),
                 type->isPointerTy() ? ABI::BufferView : ABI::Scalar,
                 type->isPointerTy()
                     ? 24u
                     : uint32_t(dl.getTypeAllocSize(type).getFixedValue())});
        }
      };
      flatten(flatten, t, 0);
      roots.push_back(std::move(root));
    }
    if (invalid)
      return createStringError(
          inconvertibleErrorCode(),
          "OpenCL: aggregate argument exceeds supported layout limits");
    if (!aggregate) {
      abis[f] = std::move(roots);
      continue;
    }
    if (!f->getReturnType()->isVoidTy())
      return createStringError(inconvertibleErrorCode(),
                               "OpenCL: kernel must return void");
    auto *wrapper =
        Function::Create(FunctionType::get(f->getReturnType(), types, false),
                         f->getLinkage(), "", module);
    wrapper->takeName(f);
    wrapper->setCallingConv(CallingConv::SPIR_KERNEL);
    // Pointer leaves may be null (optional inputs or type-only fusion views).
    // A real allocation is validated by the HAL when a non-null view is bound.
    f->setCallingConv(CallingConv::C);
    f->setLinkage(GlobalValue::InternalLinkage);
    IRBuilder<> builder(
        BasicBlock::Create(module.getContext(), "entry", wrapper));
    SmallVector<Value *> arguments;
    unsigned leaf = 0;
    for (auto [index, t] : enumerate(originals)) {
      auto construct = [&](auto &self, Type *type) -> Value * {
        if (!type->isAggregateType())
          return wrapper->getArg(leaf++);
        Value *v = PoisonValue::get(type);
        unsigned count = isa<StructType>(type)
                             ? cast<StructType>(type)->getNumElements()
                             : cast<ArrayType>(type)->getNumElements();
        for (unsigned i = 0; i < count; ++i) {
          Type *element = isa<StructType>(type)
                              ? cast<StructType>(type)->getElementType(i)
                              : cast<ArrayType>(type)->getElementType();
          v = builder.CreateInsertValue(v, self(self, element), {i});
        }
        return v;
      };
      Value *value = construct(construct, t);
      if (f->getArg(index)->hasByValAttr()) {
        auto *storage = builder.CreateAlloca(t);
        builder.CreateStore(value, storage);
        value = storage;
      }
      arguments.push_back(value);
    }
    auto *call = builder.CreateCall(f, arguments);
    call->setAttributes(f->getAttributes());
    builder.CreateRetVoid();
    InlineFunctionInfo info;
    if (!InlineFunction(*call, info).isSuccess())
      return createStringError(
          inconvertibleErrorCode(),
          "OpenCL: could not lower aggregate kernel entry");
    f->eraseFromParent();
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    PassBuilder pb;
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    FunctionPassManager pm;
    pm.addPass(SROAPass(SROAOptions::PreserveCFG));
    pm.addPass(InstCombinePass());
    pm.addPass(SimplifyCFGPass());
    pm.run(*wrapper, fam);
    abis[wrapper] = std::move(roots);
  }
  // Tensor engine qualification may be represented as ptr->i64->ptr across
  // LLVM default/global spaces. Both map to OpenCL global memory. Canonicalize
  // only exact, full-width round trips; arithmetic and other address spaces
  // remain subject to the emitter's fail-closed checks.
  for (auto &function : module) {
    SmallVector<IntToPtrInst *> casts;
    for (auto &instruction : instructions(function))
      if (auto *cast = dyn_cast<IntToPtrInst>(&instruction))
        casts.push_back(cast);
    for (auto *cast : casts) {
      auto *integer = dyn_cast<PtrToIntInst>(cast->getOperand(0));
      if (!integer || !integer->getType()->isIntegerTy(64) ||
          integer->getPointerOperand()->getType()->getPointerAddressSpace() >
              1 ||
          cast->getType()->getPointerAddressSpace() > 1)
        continue;
      IRBuilder<> builder(cast);
      Value *replacement = integer->getPointerOperand();
      if (replacement->getType() != cast->getType())
        replacement = builder.CreateAddrSpaceCast(replacement, cast->getType());
      cast->replaceAllUsesWith(replacement);
      cast->eraseFromParent();
      if (integer->use_empty())
        integer->eraseFromParent();
    }
  }
  return Error::success();
}

class Emitter {
  Module &module;
  DenseMap<const Value *, std::string> names;
  DenseMap<const BasicBlock *, std::string> labels;
  DenseMap<const Function *, std::string> functions;
  SmallPtrSet<const Function *, 8> errorFunctions;
  SmallPtrSet<const Function *, 8> barrierFunctions;
  MapVector<Type *, std::string> floatingPairs;
  MapVector<StructType *, std::string> numericRecords;
  std::string error;
  std::string code;
  raw_string_ostream out;
  unsigned next = 0;
  bool usesHalf = false;
  bool usesAtomics = false;
  bool usesInt64Atomics = false;
  bool usesAcquireRelease = false;
  bool usesSequentialConsistency = false;
  std::string atomicOrder(AtomicOrdering ordering) {
    usesAtomics = true;
    switch (ordering) {
    case AtomicOrdering::Unordered:
    case AtomicOrdering::Monotonic:
      return "memory_order_relaxed";
    case AtomicOrdering::Acquire:
      usesAcquireRelease = true;
      return "memory_order_acquire";
    case AtomicOrdering::Release:
      usesAcquireRelease = true;
      return "memory_order_release";
    case AtomicOrdering::AcquireRelease:
      usesAcquireRelease = true;
      return "memory_order_acq_rel";
    case AtomicOrdering::SequentiallyConsistent:
      usesSequentialConsistency = true;
      return "memory_order_seq_cst";
    default:
      return fail("OpenCL: invalid atomic ordering");
    }
  }
  std::string atomicScope(SyncScope::ID scope) {
    if (scope == SyncScope::SingleThread)
      return "memory_scope_work_item";
    // This ABI only exposes non-SVM cl_mem buffers and per-workgroup local
    // memory. No host or peer can concurrently address these allocations;
    // device scope therefore covers all agents in the accessible system.
    if (scope == SyncScope::System)
      return "memory_scope_device";
    SmallVector<StringRef> names;
    module.getContext().getSyncScopeNames(names);
    if (scope >= names.size())
      return fail("OpenCL: unknown atomic synchronization scope");
    if (names[scope] == "device" || names[scope] == "agent")
      return "memory_scope_device";
    if (names[scope] == "workgroup" || names[scope] == "cta")
      return "memory_scope_work_group";
    return fail("OpenCL: unsupported atomic synchronization scope");
  }
  std::string atomicPointer(const Value *pointer, Type *element,
                            bool sign = false) {
    if ((!element->isIntegerTy(32) && !element->isIntegerTy(64)) ||
        (pointer->getType()->getPointerAddressSpace() != 0 &&
         pointer->getType()->getPointerAddressSpace() != 1 &&
         pointer->getType()->getPointerAddressSpace() != 3))
      return fail(
          "OpenCL: atomics require 32/64-bit integers in global/local memory");
    usesInt64Atomics |= element->isIntegerTy(64);
    return "((volatile " + addressSpace(pointer->getType()) + " atomic_" +
           type(element, sign) + "*)" + value(pointer) + ")";
  }
  const Function *currentFunction = nullptr;
  DenseMap<const Function *, SmallVector<const GlobalVariable *>> sharedGlobals;
  bool usesSubgroups(const Function &function,
                     SmallPtrSetImpl<const Function *> &visited) {
    if (!visited.insert(&function).second)
      return false;
    if (function.getName() == "__mojo_opencl_shuffle_idx_u32_w64" ||
        function.getName() == "__mojo_opencl_subgroup_local_id" ||
        function.getName() == "__mojo_opencl_subgroup_barrier_w64")
      return true;
    for (auto &block : function)
      for (auto &instruction : block)
        if (auto *call = dyn_cast<CallBase>(&instruction))
          if (auto *callee = call->getCalledFunction())
            if (usesSubgroups(*callee, visited))
              return true;
    return false;
  }
  bool dynamicShared(const Function &function) const {
    auto found = sharedGlobals.find(&function);
    if (found == sharedGlobals.end())
      return false;
    for (auto *global : found->second)
      if (module.getDataLayout()
              .getTypeAllocSize(global->getValueType())
              .getFixedValue() == 0)
        return true;
    return false;
  }
  std::string addressSpace(Type *t) {
    switch (t->getPointerAddressSpace()) {
    case 0:
    case 1:
      return "__global";
    case 3:
      return "__local";
    case 5:
      return "__private";
    default:
      return fail("OpenCL: unsupported pointer address spaces");
    }
  }
  bool kernel(const Function &f) const {
    return f.getCallingConv() == CallingConv::SPIR_KERNEL ||
           f.hasFnAttribute("mojo.opencl.kernel");
  }
  std::string fail(const Twine &message) {
    if (error.empty())
      error = message.str();
    return "/* rejected */0";
  }
  std::string memoryPointerType(Type *element, Type *storagePointer) {
    // A pointer value and the storage holding that value have independent
    // address spaces: __global uchar * __private *, not two qualifiers on
    // the same pointee type.
    auto storageSpace = addressSpace(storagePointer);
    if (element->isPointerTy())
      return type(element) + " " + storageSpace + " *";
    return storageSpace + " " + type(element) + " *";
  }
  static bool isSinCos(const Value *value) {
#if LLVM_VERSION_MAJOR >= 23
    auto *intrinsic = dyn_cast<IntrinsicInst>(value);
    return intrinsic && intrinsic->getIntrinsicID() == Intrinsic::sincos;
#else
    return false;
#endif
  }
  std::string fieldName(StructType *record, unsigned index) {
    if (!record->isPacked() && record->getNumElements() == 2) {
      auto *first = record->getElementType(0);
      auto *second = record->getElementType(1);
      if (first == second && (first->getScalarType()->isFloatTy() ||
                              first->getScalarType()->isHalfTy()))
        return index ? "second" : "first";
      if (second->isIntegerTy(1) &&
          (first->isIntegerTy(32) || first->isIntegerTy(64)))
        return index ? "success" : "old";
    }
    return "field_" + std::to_string(index);
  }
  std::string type(Type *t, bool isSigned = false) {
    if (auto *st = dyn_cast<StructType>(t)) {
      if (!st->isPacked() && st->getNumElements() == 2 &&
          st->getElementType(0) == st->getElementType(1) &&
          (st->getElementType(0)->getScalarType()->isFloatTy() ||
           st->getElementType(0)->getScalarType()->isHalfTy())) {
        auto name = type(st->getElementType(0));
        floatingPairs[st->getElementType(0)] = name;
        return "mojo_pair_" + name;
      }
      if (!st->isPacked() && st->getNumElements() == 2 &&
          st->getElementType(1)->isIntegerTy(1) &&
          (st->getElementType(0)->isIntegerTy(32) ||
           st->getElementType(0)->isIntegerTy(64)))
        return st->getElementType(0)->isIntegerTy(32) ? "mojo_cas32"
                                                      : "mojo_cas64";
      // Optimized reduction monoids can retain numeric record values in SSA
      // (loads, phis, helper calls). Keep LLVM's layout explicitly rather than
      // assuming the vendor C compiler chooses the same padding.
      if (!st->isOpaque() && st->getNumElements() &&
          llvm::all_of(st->elements(), [](Type *field) {
            return field->isIntegerTy() || field->isFloatTy() ||
                   field->isHalfTy();
          })) {
        for (auto *field : st->elements())
          type(field);
        auto [it, inserted] = numericRecords.try_emplace(
            st, "mojo_record_" + std::to_string(numericRecords.size()));
        return it->second;
      }
    }
    if (t->isVoidTy())
      return "void";
    if (t->isFloatTy())
      return "float";
    if (t->isHalfTy()) {
      usesHalf = true;
      return "half";
    }
    if (auto *vector = dyn_cast<FixedVectorType>(t)) {
      unsigned width = vector->getNumElements();
      // LLVM uses one-lane vectors for scalar loads broadcast into wider
      // vectors. Their scalar representation has identical size/alignment.
      if (width == 1)
        return type(vector->getElementType(), isSigned);
      if (width != 2 && width != 3 && width != 4 && width != 8 && width != 16)
        return fail("OpenCL: unsupported LLVM vector width " +
                    std::to_string(width) + " (expected 2, 3, 4, 8 or 16)");
      if (vector->getElementType()->isPointerTy())
        return fail("OpenCL: vectors of pointers are unsupported");
      return type(vector->getElementType(), isSigned) + std::to_string(width);
    }
    if (t->isPointerTy()) {
      return addressSpace(t) + " uchar *";
    }
    if (t->isIntegerTy()) {
      unsigned width = t->getIntegerBitWidth();
      if (width == 1)
        return isSigned ? "char" : "uchar";
      if (width == 8)
        return isSigned ? "char" : "uchar";
      if (width == 16)
        return isSigned ? "short" : "ushort";
      if (width == 32)
        return isSigned ? "int" : "uint";
      if (width == 64)
        return isSigned ? "long" : "ulong";
    }
    std::string description;
    raw_string_ostream stream(description);
    t->print(stream);
    return fail("OpenCL: unsupported LLVM type " + description);
  }
  std::string value(const Value *v) {
    if (auto *global = dyn_cast<GlobalVariable>(v)) {
      if (global->getAddressSpace() != 3 || !currentFunction ||
          !kernel(*currentFunction))
        return fail("OpenCL: shared globals must be used directly in a kernel");
    }
    if (auto *constant = dyn_cast<ConstantExpr>(v)) {
      auto *instruction = constant->getAsInstruction();
      auto result = expression(*instruction);
      instruction->deleteValue();
      return result;
    }
    if (isa<PoisonValue>(v) || isa<UndefValue>(v) ||
        isa<ConstantAggregateZero>(v)) {
      if (v->getType()->isStructTy())
        return "(" + type(v->getType()) + "){0}";
      return "((" + type(v->getType()) + ")0)";
    }
    if (auto *record = dyn_cast<ConstantStruct>(v)) {
      auto *ty = record->getType();
      std::string result = "(" + type(ty) + "){";
      for (unsigned field = 0; field < record->getNumOperands(); ++field) {
        if (field)
          result += ",";
        result +=
            "." + fieldName(ty, field) + "=" + value(record->getOperand(field));
      }
      return result + "}";
    }
    if (auto *vector = dyn_cast<FixedVectorType>(v->getType()))
      if (auto *constant = dyn_cast<Constant>(v)) {
        std::string result = "((" + type(vector) + ")(";
        for (unsigned lane = 0; lane < vector->getNumElements(); ++lane) {
          if (lane)
            result += ",";
          result += value(constant->getAggregateElement(lane));
        }
        return result + "))";
      }
    if (auto *integer = dyn_cast<ConstantInt>(v)) {
      SmallString<32> number;
      integer->getValue().toString(number, 16, false);
      return "((" + type(v->getType()) + ")0x" + number.str().str() + "UL)";
    }
    if (auto *fp = dyn_cast<ConstantFP>(v)) {
      if (!v->getType()->isFloatTy() && !v->getType()->isHalfTy())
        return fail("OpenCL: only f16/f32 floating constants are supported");
      SmallString<32> bits;
      fp->getValueAPF().bitcastToAPInt().toString(bits, 16, false);
      return v->getType()->isHalfTy()
                 ? "as_half((ushort)0x" + bits.str().str() + "U)"
                 : "as_float((uint)0x" + bits.str().str() + "U)";
    }
    if (isa<ConstantPointerNull>(v))
      return "((__global uchar *)0)";
    if (auto found = names.find(v); found != names.end())
      return found->second;
    return fail("OpenCL: unsupported constant, global or undefined value");
  }
  std::string signedValue(const Value *v) {
    if (auto *vector = dyn_cast<FixedVectorType>(v->getType())) {
      if (vector->getElementType()->isIntegerTy(1))
        return "(-convert_int" +
               (vector->getNumElements() == 1
                    ? ""
                    : std::to_string(vector->getNumElements())) +
               "(" + value(v) + "))";
      return "as_" + type(v->getType(), true) + "(" + value(v) + ")";
    }
    if (v->getType()->isIntegerTy(1))
      return "(-((int)" + value(v) + "))";
    return "((" + type(v->getType(), true) + ")" + value(v) + ")";
  }
  std::string component(const Value *vector, unsigned lane) {
    if (cast<FixedVectorType>(vector->getType())->getNumElements() == 1)
      return value(vector);
    static constexpr char digits[] = "0123456789abcdef";
    return "(" + value(vector) + ").s" + digits[lane];
  }
  std::string args(CallBase &call) {
    std::string result;
    for (const auto &arg : call.args()) {
      if (!result.empty())
        result += ", ";
      result += value(arg.get());
    }
    return result;
  }
  std::string reduction(CallBase &call, Intrinsic::ID id) {
    auto *vector = call.getArgOperand(call.arg_size() - 1);
    auto *vectorType = dyn_cast<FixedVectorType>(vector->getType());
    if (!vectorType)
      return fail("OpenCL: reduction requires a fixed vector");
    bool ordered = id == Intrinsic::vector_reduce_fadd ||
                   id == Intrinsic::vector_reduce_fmul;
    std::string result =
        ordered ? value(call.getArgOperand(0)) : component(vector, 0);
    for (unsigned lane = ordered ? 0 : 1; lane < vectorType->getNumElements();
         ++lane) {
      std::string rhs = component(vector, lane), op;
      switch (id) {
      case Intrinsic::vector_reduce_add:
      case Intrinsic::vector_reduce_fadd:
        op = "+";
        break;
      case Intrinsic::vector_reduce_mul:
      case Intrinsic::vector_reduce_fmul:
        op = "*";
        break;
      case Intrinsic::vector_reduce_and:
        op = "&";
        break;
      case Intrinsic::vector_reduce_or:
        op = "|";
        break;
      case Intrinsic::vector_reduce_xor:
        op = "^";
        break;
      default: {
        bool maximum = id == Intrinsic::vector_reduce_fmax ||
                       id == Intrinsic::vector_reduce_smax ||
                       id == Intrinsic::vector_reduce_umax;
        bool sign = id == Intrinsic::vector_reduce_smax ||
                    id == Intrinsic::vector_reduce_smin;
        if (sign) {
          result = "((" + type(call.getType(), true) + ")" + result + ")";
          rhs = "((" + type(call.getType(), true) + ")" + rhs + ")";
        }
        result = std::string(call.getType()->isFloatingPointTy() ? "f" : "") +
                 (maximum ? "max(" : "min(") + result + "," + rhs + ")";
        continue;
      }
      }
      if (call.getType()->isIntegerTy() &&
          call.getType()->getIntegerBitWidth() < 32) {
        result = "((uint)" + result + ")";
        rhs = "((uint)" + rhs + ")";
      }
      result = "((" + type(call.getType()) + ")(" + result + op + rhs + "))";
      if (call.getType()->isIntegerTy(1))
        result = "(" + result + " & 1)";
    }
    return result;
  }
  std::string call(CallBase &call) {
    const Function *fn = call.getCalledFunction();
    if (!fn || call.isInlineAsm())
      return fail("OpenCL: indirect calls and inline assembly are unsupported");
    StringRef name = fn->getName();
    if (name == "__mojo_opencl_shuffle_idx_u32_w64") {
      if (call.arg_size() != 2 || !call.getType()->isIntegerTy(32) ||
          !call.getArgOperand(0)->getType()->isIntegerTy(32) ||
          !call.getArgOperand(1)->getType()->isIntegerTy(32))
        return fail("OpenCL: invalid indexed subgroup shuffle signature");
      return "sub_group_shuffle(" + args(call) + ")";
    }
    if (name == "__mojo_opencl_subgroup_local_id") {
      if (call.arg_size() || !call.getType()->isIntegerTy(64))
        return fail("OpenCL: invalid subgroup ID signature");
      return "get_sub_group_local_id()";
    }
    if (name == "__mojo_opencl_subgroup_barrier_w64") {
      if (call.arg_size() || !call.getType()->isVoidTy())
        return fail("OpenCL: invalid subgroup barrier signature");
      return "sub_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)";
    }
    if (name == "__mojo_opencl_barrier") {
      if (call.arg_size() || !call.getType()->isVoidTy())
        return fail("OpenCL: invalid workgroup barrier signature");
      return "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)";
    }
    for (auto builtin :
         {"get_local_id", "get_group_id", "get_local_size", "get_num_groups"}) {
      std::string prefix = "__mojo_opencl_" + std::string(builtin) + "_";
      if (name.starts_with(prefix) && name.size() == prefix.size() + 1) {
        if (call.arg_size() || !call.getType()->isIntegerTy(64))
          return fail("OpenCL: invalid work-item builtin signature");
        char dim = name.back();
        if (dim < 'x' || dim > 'z')
          return fail("OpenCL: invalid builtin dimension");
        return std::string(builtin) + "(" + std::to_string(dim - 'x') + ")";
      }
    }
    if (fn->isIntrinsic()) {
      switch (fn->getIntrinsicID()) {
      case Intrinsic::vector_reduce_add:
      case Intrinsic::vector_reduce_mul:
      case Intrinsic::vector_reduce_and:
      case Intrinsic::vector_reduce_or:
      case Intrinsic::vector_reduce_xor:
      case Intrinsic::vector_reduce_smax:
      case Intrinsic::vector_reduce_smin:
      case Intrinsic::vector_reduce_umax:
      case Intrinsic::vector_reduce_umin:
      case Intrinsic::vector_reduce_fadd:
      case Intrinsic::vector_reduce_fmul:
      case Intrinsic::vector_reduce_fmax:
      case Intrinsic::vector_reduce_fmin:
        return reduction(call, fn->getIntrinsicID());
      case Intrinsic::smax:
      case Intrinsic::smin:
      case Intrinsic::umax:
      case Intrinsic::umin: {
        bool sign = fn->getIntrinsicID() == Intrinsic::smax ||
                    fn->getIntrinsicID() == Intrinsic::smin;
        bool maximum = fn->getIntrinsicID() == Intrinsic::smax ||
                       fn->getIntrinsicID() == Intrinsic::umax;
        auto lhs = sign ? signedValue(call.getArgOperand(0))
                        : value(call.getArgOperand(0));
        auto rhs = sign ? signedValue(call.getArgOperand(1))
                        : value(call.getArgOperand(1));
        auto result =
            std::string(maximum ? "max(" : "min(") + lhs + "," + rhs + ")";
        if (sign && call.getType()->isVectorTy())
          result = "as_" + type(call.getType()) + "(" + result + ")";
        return result;
      }
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
        return "((void)0)";
      case Intrinsic::ctpop:
        return "popcount(" + value(call.getArgOperand(0)) + ")";
      case Intrinsic::ctlz:
        return "clz(" + value(call.getArgOperand(0)) + ")";
      case Intrinsic::cttz: {
        // popcount(~x & (x - 1)) also handles zero and is available in C 1.2.
        auto x = value(call.getArgOperand(0));
        auto t = type(call.getType());
        return "popcount((" + t + ")((~" + x + ") & (" + t + ")(" + x + "-(" +
               t + ")1)))";
      }
      case Intrinsic::trunc:
        return "trunc(" + args(call) + ")";
      case Intrinsic::rint:
      case Intrinsic::nearbyint:
        return "rint(" + args(call) + ")";
      case Intrinsic::round:
        return "round(" + args(call) + ")";
      case Intrinsic::copysign:
        return "copysign(" + args(call) + ")";
      case Intrinsic::pow:
        return "pow(" + args(call) + ")";
      case Intrinsic::log10:
        return "log10(" + args(call) + ")";
      case Intrinsic::fabs:
        return "fabs(" + args(call) + ")";
      case Intrinsic::maxnum:
        return "fmax(" + args(call) + ")";
      case Intrinsic::minnum:
        return "fmin(" + args(call) + ")";
      case Intrinsic::sqrt:
        return "sqrt(" + args(call) + ")";
      case Intrinsic::exp:
        return "exp(" + args(call) + ")";
      case Intrinsic::exp2:
        return "exp2(" + args(call) + ")";
      case Intrinsic::log:
        return "log(" + args(call) + ")";
      case Intrinsic::log2:
        return "log2(" + args(call) + ")";
      case Intrinsic::sin:
        return "sin(" + args(call) + ")";
      case Intrinsic::cos:
        return "cos(" + args(call) + ")";
      case Intrinsic::floor:
        return "floor(" + args(call) + ")";
      case Intrinsic::ceil:
        return "ceil(" + args(call) + ")";
      case Intrinsic::fma:
        // AFN explicitly permits an approximate implementation. Keep strict
        // fma correctly rounded; OpenCL mad is a different numerical contract.
        return std::string(call.hasApproxFunc() ? "mad(" : "fma(") +
               args(call) + ")";
      case Intrinsic::fmuladd:
        return "fma(" + args(call) + ")";
      default:
        return fail("OpenCL: unsupported intrinsic '" + name + "'");
      }
    }
    if (!fn->isDeclaration()) {
      if (kernel(*fn))
        return fail("OpenCL: kernels cannot call another kernel");
      auto arguments = args(call);
      if (errorFunctions.contains(fn))
        arguments += (arguments.empty() ? "" : ",") + std::string("mojo_error");
      return functions[fn] + "(" + arguments + ")";
    }
    return fail("OpenCL: unsupported external call '" + name + "'");
  }
  void edge(BasicBlock *from, BasicBlock *to) {
    // Phi inputs are evaluated in parallel, including loop-carried swaps.
    for (auto &phi : to->phis())
      out << names[&phi]
          << "_next = " << value(phi.getIncomingValueForBlock(from)) << ";\n";
    for (auto &phi : to->phis())
      out << names[&phi] << " = " << names[&phi] << "_next;\n";
    out << "goto " << labels[to] << ";\n";
  }
  std::string binary(BinaryOperator &i) {
    bool sign = i.getOpcode() == Instruction::SDiv ||
                i.getOpcode() == Instruction::SRem ||
                i.getOpcode() == Instruction::AShr;
    auto lhs = sign ? signedValue(i.getOperand(0)) : value(i.getOperand(0));
    auto rhs =
        i.getOpcode() == Instruction::SDiv || i.getOpcode() == Instruction::SRem
            ? signedValue(i.getOperand(1))
            : value(i.getOperand(1));
    // OpenCL promotes uchar/ushort to signed int. Force unsigned arithmetic
    // before narrow multiplication/left shifts to preserve LLVM's wrapping.
    if (!sign && i.getType()->isIntegerTy() &&
        i.getType()->getIntegerBitWidth() < 32) {
      lhs = "((uint)" + lhs + ")";
      rhs = "((uint)" + rhs + ")";
    }
    const char *op = nullptr;
    switch (i.getOpcode()) {
    case Instruction::Add:
    case Instruction::FAdd:
      op = "+";
      break;
    case Instruction::Sub:
    case Instruction::FSub:
      op = "-";
      break;
    case Instruction::Mul:
    case Instruction::FMul:
      op = "*";
      break;
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
      op = "/";
      break;
    case Instruction::URem:
    case Instruction::SRem:
      op = "%";
      break;
    case Instruction::FRem:
      return "fmod(" + lhs + "," + rhs + ")";
    case Instruction::And:
      op = "&";
      break;
    case Instruction::Or:
      op = "|";
      break;
    case Instruction::Xor:
      op = "^";
      break;
    case Instruction::Shl:
      op = "<<";
      break;
    case Instruction::LShr:
    case Instruction::AShr:
      op = ">>";
      break;
    default:
      return fail("OpenCL: unsupported binary operation");
    }
    auto expression = "(" + lhs + op + rhs + ")";
    if (i.getType()->getScalarType()->isIntegerTy(1))
      expression = "(" + expression + " & 1)";
    if (sign && i.getType()->isVectorTy())
      expression = "as_" + type(i.getType()) + "(" + expression + ")";
    return expression;
  }
  std::string compare(CmpInst &i) {
    auto lhs = value(i.getOperand(0)), rhs = value(i.getOperand(1));
    if (i.isIntPredicate()) {
      if (i.isSigned()) {
        lhs = signedValue(i.getOperand(0));
        rhs = signedValue(i.getOperand(1));
      }
      const char *op = nullptr;
      switch (i.getPredicate()) {
      case CmpInst::ICMP_EQ:
        op = "==";
        break;
      case CmpInst::ICMP_NE:
        op = "!=";
        break;
      case CmpInst::ICMP_UGT:
      case CmpInst::ICMP_SGT:
        op = ">";
        break;
      case CmpInst::ICMP_UGE:
      case CmpInst::ICMP_SGE:
        op = ">=";
        break;
      case CmpInst::ICMP_ULT:
      case CmpInst::ICMP_SLT:
        op = "<";
        break;
      case CmpInst::ICMP_ULE:
      case CmpInst::ICMP_SLE:
        op = "<=";
        break;
      default:
        return fail("OpenCL: unsupported integer predicate");
      }
      return "(" + lhs + op + rhs + ")";
    }
    std::string nan = "(isnan(" + lhs + ") || isnan(" + rhs + "))";
    if (i.getPredicate() == CmpInst::FCMP_FALSE)
      return "0";
    if (i.getPredicate() == CmpInst::FCMP_TRUE)
      return "1";
    if (i.getPredicate() == CmpInst::FCMP_ORD)
      return "(!" + nan + ")";
    if (i.getPredicate() == CmpInst::FCMP_UNO)
      return nan;
    const char *op = nullptr;
    switch (i.getPredicate()) {
    case CmpInst::FCMP_OEQ:
    case CmpInst::FCMP_UEQ:
      op = "==";
      break;
    case CmpInst::FCMP_ONE:
    case CmpInst::FCMP_UNE:
      op = "!=";
      break;
    case CmpInst::FCMP_OGT:
    case CmpInst::FCMP_UGT:
      op = ">";
      break;
    case CmpInst::FCMP_OGE:
    case CmpInst::FCMP_UGE:
      op = ">=";
      break;
    case CmpInst::FCMP_OLT:
    case CmpInst::FCMP_ULT:
      op = "<";
      break;
    case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_ULE:
      op = "<=";
      break;
    default:
      return fail("OpenCL: unsupported floating predicate");
    }
    bool unordered = i.getPredicate() >= CmpInst::FCMP_UEQ;
    return "(" + std::string(unordered ? nan + " || " : "!" + nan + " && ") +
           "(" + lhs + op + rhs + "))";
  }
  uint64_t nativeVectorAlignment(FixedVectorType *vector) {
    // OpenCL pads a three-lane vector to four lanes. Wider vectors can have
    // stricter alignment than the borrowed LLVM data layout; never infer this
    // from that layout's ABI alignment.
    unsigned lanes = vector->getNumElements();
    return (lanes == 3 ? 4 : lanes) *
           module.getDataLayout()
               .getTypeStoreSize(vector->getElementType())
               .getFixedValue();
  }
  std::string bytePointer(const Value *pointer, uint64_t offset = 0) {
    return "((" + addressSpace(pointer->getType()) + " uchar*)" +
           value(pointer) + "+" + std::to_string(offset) + ")";
  }
  std::string unalignedLoad(Type *ty, const Value *pointer) {
    auto *scalar = ty->getScalarType();
    if (!scalar->isIntOrIntVectorTy() && !scalar->isFloatingPointTy())
      return fail("OpenCL: unaligned nonnumeric loads are unsupported");
    unsigned bytes = scalar->getPrimitiveSizeInBits() / 8;
    if (bytes != 2 && bytes != 4 && bytes != 8)
      return fail("OpenCL: unsupported unaligned element size");
    auto *vector = dyn_cast<FixedVectorType>(ty);
    unsigned lanes = vector ? vector->getNumElements() : 1;
    std::string result = vector ? "((" + type(ty) + ")(" : "";
    for (unsigned lane = 0; lane < lanes; ++lane) {
      if (lane)
        result += ",";
      result += "as_" + type(scalar) + "(vload" + std::to_string(bytes) +
                "(0," + bytePointer(pointer, lane * bytes) + "))";
    }
    return result + (vector ? "))" : "");
  }
  void unalignedStore(raw_ostream &out, const Value *operand,
                      const Value *pointer) {
    auto *scalar = operand->getType()->getScalarType();
    if (!scalar->isIntOrIntVectorTy() && !scalar->isFloatingPointTy()) {
      fail("OpenCL: unaligned nonnumeric stores are unsupported");
      return;
    }
    unsigned bytes = scalar->getPrimitiveSizeInBits() / 8;
    if (bytes != 2 && bytes != 4 && bytes != 8) {
      fail("OpenCL: unsupported unaligned element size");
      return;
    }
    auto *vector = dyn_cast<FixedVectorType>(operand->getType());
    unsigned lanes = vector ? vector->getNumElements() : 1;
    for (unsigned lane = 0; lane < lanes; ++lane)
      out << "vstore" << bytes << "(as_uchar" << bytes << "("
          << (vector ? component(operand, lane) : value(operand)) << "),0,"
          << bytePointer(pointer, lane * bytes) << ");\n";
  }
  std::string expression(Instruction &i) {
    if (auto *freeze = dyn_cast<FreezeInst>(&i))
      return value(freeze->getOperand(0));
    if (auto *a = dyn_cast<AllocaInst>(&i))
      return names[a] + "_storage";
    if (auto *extract = dyn_cast<ExtractValueInst>(&i)) {
      if (auto *record =
              dyn_cast<StructType>(extract->getAggregateOperand()->getType()))
        if (extract->getNumIndices() == 1 &&
            extract->getIndices()[0] < record->getNumElements()) {
          type(record);
          return "(" + value(extract->getAggregateOperand()) + ")." +
                 fieldName(record, extract->getIndices()[0]);
        }
      return fail("OpenCL: unsupported aggregate extraction");
    }
    if (auto *atomic = dyn_cast<AtomicRMWInst>(&i)) {
      if (atomic->isVolatile() ||
          atomic->getAlign() <
              module.getDataLayout().getABITypeAlign(atomic->getType()))
        return fail(
            "OpenCL: volatile or unaligned atomic operations are unsupported");
      std::string op;
      bool sign = false;
      switch (atomic->getOperation()) {
      case AtomicRMWInst::Xchg:
        op = "atomic_exchange_explicit";
        break;
      case AtomicRMWInst::Add:
        op = "atomic_fetch_add_explicit";
        break;
      case AtomicRMWInst::Sub:
        op = "atomic_fetch_sub_explicit";
        break;
      case AtomicRMWInst::And:
        op = "atomic_fetch_and_explicit";
        break;
      case AtomicRMWInst::Or:
        op = "atomic_fetch_or_explicit";
        break;
      case AtomicRMWInst::Xor:
        op = "atomic_fetch_xor_explicit";
        break;
      case AtomicRMWInst::Max:
        sign = true;
        [[fallthrough]];
      case AtomicRMWInst::UMax:
        op = "atomic_fetch_max_explicit";
        break;
      case AtomicRMWInst::Min:
        sign = true;
        [[fallthrough]];
      case AtomicRMWInst::UMin:
        op = "atomic_fetch_min_explicit";
        break;
      default:
        return fail("OpenCL: unsupported atomic read-modify-write operation");
      }
      return op + "(" +
             atomicPointer(atomic->getPointerOperand(), atomic->getType(),
                           sign) +
             "," +
             (sign ? signedValue(atomic->getValOperand())
                   : value(atomic->getValOperand())) +
             "," + atomicOrder(atomic->getOrdering()) + "," +
             atomicScope(atomic->getSyncScopeID()) + ")";
    }
    if (auto *fence = dyn_cast<FenceInst>(&i))
      return "atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | "
             "CLK_LOCAL_MEM_FENCE," +
             atomicOrder(fence->getOrdering()) + "," +
             atomicScope(fence->getSyncScopeID()) + ")";
    if (auto *op = dyn_cast<BinaryOperator>(&i))
      return binary(*op);
    if (auto *cmp = dyn_cast<CmpInst>(&i)) {
      auto result = compare(*cmp);
      if (i.getType()->isVectorTy()) {
        if (cmp->getPredicate() == CmpInst::FCMP_TRUE ||
            cmp->getPredicate() == CmpInst::FCMP_FALSE)
          return "((" + type(i.getType()) + ")" + result + ")";
        // OpenCL comparisons produce signed masks with the operand's lane
        // width. An int literal cannot be combined with a short/char vector.
        // Convert first, then normalize in the boolean representation.
        auto booleanType = type(i.getType());
        return "(convert_" + booleanType + "(" + result + ") & ((" +
               booleanType + ")1))";
      }
      return result;
    }
    if (auto *load = dyn_cast<LoadInst>(&i)) {
      if (load->getType()->isVectorTy() &&
          load->getType()->getScalarType()->isIntegerTy(1))
        return fail("OpenCL: packed boolean vector memory is unsupported");
      if (load->isAtomic()) {
        if (load->isVolatile() ||
            load->getAlign() <
                module.getDataLayout().getABITypeAlign(load->getType()))
          return fail(
              "OpenCL: volatile or unaligned atomic loads are unsupported");
        bool floating = load->getType()->isFloatTy();
        auto *element =
            floating ? Type::getInt32Ty(module.getContext()) : load->getType();
        auto result = "atomic_load_explicit(" +
                      atomicPointer(load->getPointerOperand(), element) + "," +
                      atomicOrder(load->getOrdering()) + "," +
                      atomicScope(load->getSyncScopeID()) + ")";
        return floating ? "as_float(" + result + ")" : result;
      }
      if (load->isVolatile())
        return fail("OpenCL: volatile loads are unsupported");
      if (load->getAlign() < module.getDataLayout().getABITypeAlign(
                                 load->getType()->getScalarType()))
        return unalignedLoad(load->getType(), load->getPointerOperand());
      if (auto *vector = dyn_cast<FixedVectorType>(i.getType());
          vector && load->getAlign().value() >= nativeVectorAlignment(vector))
        return "*((" + addressSpace(load->getPointerOperand()->getType()) +
               " " + type(i.getType()) + "*)" +
               value(load->getPointerOperand()) + ")";
      if (auto *vector = dyn_cast<FixedVectorType>(i.getType()))
        return "vload" + std::to_string(vector->getNumElements()) + "(0, (" +
               addressSpace(load->getPointerOperand()->getType()) + " " +
               type(vector->getElementType()) + "*)" +
               value(load->getPointerOperand()) + ")";
      return "*((" +
             memoryPointerType(i.getType(),
                               load->getPointerOperand()->getType()) +
             ")" + value(load->getPointerOperand()) + ")";
    }
    if (auto *gep = dyn_cast<GetElementPtrInst>(&i)) {
      SmallMapVector<Value *, APInt, 4> variables;
      APInt constant(64, 0);
      if (!gep->collectOffset(module.getDataLayout(), 64, variables, constant))
        return fail("OpenCL: cannot represent pointer offset");
      SmallString<32> number;
      constant.toString(number, 16, false);
      // Preserve the pointer's address-space provenance. Integer round trips
      // make Adreno's OpenCL compiler lose inter-work-item aliasing for local
      // memory, even across a workgroup barrier. Compute the byte displacement
      // modulo the LLVM index width, then add it to the byte pointer directly.
      std::string result = "((" + type(gep->getType()) + ")(" +
                           value(gep->getPointerOperand()) + "+((long)(0x" +
                           number.str().str() + "UL";
      for (auto &entry : variables) {
        SmallString<32> scale;
        entry.second.toString(scale, 16, false);
        result += "+((ulong)" + signedValue(entry.first) + ")*0x" +
                  scale.str().str() + "UL";
      }
      return result + "))))";
    }
    if (auto *extract = dyn_cast<ExtractElementInst>(&i)) {
      auto *vector = extract->getVectorOperand();
      unsigned width =
          cast<FixedVectorType>(vector->getType())->getNumElements();
      if (auto *index = dyn_cast<ConstantInt>(extract->getIndexOperand()))
        return index->getZExtValue() < width
                   ? component(vector, unsigned(index->getZExtValue()))
                   : "0";
      std::string result = "0";
      for (unsigned lane = 0; lane < width; ++lane)
        result = "(" + value(extract->getIndexOperand()) +
                 " == " + std::to_string(lane) + " ? " +
                 component(vector, lane) + " : " + result + ")";
      return result;
    }
    if (auto *shuffle = dyn_cast<ShuffleVectorInst>(&i)) {
      auto *input = cast<FixedVectorType>(shuffle->getOperand(0)->getType());
      std::string result = "((" + type(i.getType()) + ")(";
      for (unsigned lane = 0; lane < shuffle->getShuffleMask().size(); ++lane) {
        if (lane)
          result += ",";
        int index = shuffle->getMaskValue(lane);
        if (index < 0)
          result += "0";
        else {
          unsigned width = input->getNumElements();
          result += component(shuffle->getOperand(index < int(width) ? 0 : 1),
                              unsigned(index) % width);
        }
      }
      return result + "))";
    }
    if (auto *select = dyn_cast<SelectInst>(&i)) {
      if (auto *vector =
              dyn_cast<FixedVectorType>(select->getCondition()->getType())) {
        auto *element = i.getType()->getScalarType();
        unsigned bits =
            element->isIntegerTy(1) ? 8 : element->getPrimitiveSizeInBits();
        std::string mask =
            type(IntegerType::get(module.getContext(), bits), true) +
            std::to_string(vector->getNumElements());
        return "select(" + value(select->getFalseValue()) + "," +
               value(select->getTrueValue()) + ", -convert_" + mask + "(" +
               value(select->getCondition()) + "))";
      }
      return "(" + value(select->getCondition()) + " ? " +
             value(select->getTrueValue()) + " : " +
             value(select->getFalseValue()) + ")";
    }
    if (auto *cast = dyn_cast<CastInst>(&i)) {
      // The OpenCL target maps both default/global LLVM pointers to OpenCL
      // global memory. Standard tensor engines make this qualification
      // explicit; it does not cross OpenCL's disjoint local/private spaces.
      if (auto *space = dyn_cast<AddrSpaceCastInst>(cast)) {
        if (space->getSrcAddressSpace() <= 1 &&
            space->getDestAddressSpace() <= 1)
          return value(space->getOperand(0));
      }
      if (isa<IntToPtrInst>(cast) || isa<PtrToIntInst>(cast) ||
          isa<AddrSpaceCastInst>(cast))
        return fail(Twine("OpenCL: unsupported pointer cast '") +
                    cast->getOpcodeName() + "'");
      if (isa<BitCastInst>(cast)) {
        if ((i.getType()->isVectorTy() &&
             i.getType()->getScalarType()->isIntegerTy(1)) ||
            (cast->getSrcTy()->isVectorTy() &&
             cast->getSrcTy()->getScalarType()->isIntegerTy(1)))
          return fail("OpenCL: packed boolean vector bitcasts are unsupported");
        if (i.getType()->isPointerTy() && cast->getSrcTy()->isPointerTy())
          return value(cast->getOperand(0));
        return "as_" + type(i.getType()) + "(" + value(cast->getOperand(0)) +
               ")";
      }
      bool sign = isa<SExtInst>(cast) || isa<SIToFPInst>(cast);
      std::string operand =
          sign ? signedValue(cast->getOperand(0)) : value(cast->getOperand(0));
      if (i.getType()->isVectorTy()) {
        if (isa<TruncInst>(cast) &&
            i.getType()->getScalarType()->isIntegerTy(1))
          operand = "(" + operand + " & 1)";
        if (isa<FPToSIInst>(cast)) {
          auto result = "as_" + type(i.getType()) + "(convert_" +
                        type(i.getType(), true) + "_rtz(" + operand + "))";
          if (i.getType()->getScalarType()->isIntegerTy(1))
            result = "(" + result + " & 1)";
          return result;
        }
        std::string rounding = isa<FPTruncInst>(cast)  ? "_rte"
                               : isa<FPToUIInst>(cast) ? "_rtz"
                                                       : "";
        return "convert_" + type(i.getType()) + rounding + "(" + operand + ")";
      }
      if (isa<TruncInst>(cast) && i.getType()->isIntegerTy(1))
        return "((uchar)(" + operand + " & 1))";
      if (isa<FPToSIInst>(cast))
        return "((" + type(i.getType()) + ")(((" + type(i.getType(), true) +
               ")" + operand + ")" +
               (i.getType()->isIntegerTy(1) ? " & 1" : "") + "))";
      return "((" + type(i.getType()) + ")" + operand + ")";
    }
    if (auto *c = dyn_cast<CallBase>(&i))
      return call(*c);
    if (i.getOpcode() == Instruction::FNeg)
      return "(-" + value(i.getOperand(0)) + ")";
    return fail(Twine("OpenCL: unsupported instruction '") + i.getOpcodeName() +
                "'");
  }
  void signature(Function &f) {
    if (kernel(f) && f.hasFnAttribute("mojo.opencl.work_group_size_hint")) {
      uint64_t hint = 0;
      if (f.getFnAttribute("mojo.opencl.work_group_size_hint")
              .getValueAsString()
              .getAsInteger(10, hint) ||
          hint == 0 || hint > UINT32_MAX)
        fail("OpenCL: invalid workgroup-size hint");
      out << "__attribute__((work_group_size_hint(" << hint << ",1,1))) ";
    }
    if (kernel(f))
      out << "__kernel ";
    else
      out << "static ";
    out << type(f.getReturnType()) << " " << functions[&f] << "(";
    bool first = true;
    for (auto &arg : f.args()) {
      if (!first)
        out << ",";
      first = false;
      out << type(arg.getType()) << " " << names[&arg];
      if (kernel(f) && arg.getType()->isPointerTy())
        out << "_base, ulong " << names[&arg] << "_offset";
    }
    if (kernel(f) && dynamicShared(f)) {
      if (!first)
        out << ",";
      out << "__local uchar *mojo_dynamic_shared";
      first = false;
    }
    if (errorFunctions.contains(&f)) {
      if (!first)
        out << ",";
      out << "__global uint *mojo_error";
      first = false;
    }
    if (first)
      out << "void";
    out << ")";
  }
  void returnFromFunction(const Function &function) {
    out << "return";
    auto *result = function.getReturnType();
    if (!result->isVoidTy())
      out << " (" << type(result) << ")"
          << (result->isAggregateType() ? "{0}" : "0");
    out << ";\n";
  }

public:
  explicit Emitter(Module &m) : module(m), out(code) {}
  AggregateABIs aggregateABIs;
  Expected<M::OpenCL::Bundle> run() {
    if (module.getTargetTriple().getOSName() != "opencl")
      return createStringError(inconvertibleErrorCode(),
                               "OpenCL: expected an OpenCL-target module");
    if (verifyModule(module, &errs()))
      return createStringError(inconvertibleErrorCode(),
                               "OpenCL: invalid LLVM IR");
    if (!module.getDataLayout().isLittleEndian() ||
        module.getDataLayout().getPointerSizeInBits(0) != 64)
      return createStringError(
          inconvertibleErrorCode(),
          "OpenCL: requires a 64-bit little-endian data layout");
    if (auto err = flattenKernelArguments(module, aggregateABIs))
      return std::move(err);
    // LLVM's generic stack address space must retain private provenance before
    // textual emission. Use the existing address-space inference machinery to
    // propagate it through GEP/select/phi and memory intrinsics.
    for (auto &f : module) {
      if (f.isDeclaration())
        continue;
      // Route aggregate SSA joins through LLVM's existing scalar replacement.
      SmallVector<PHINode *> aggregatePhis;
      for (auto &block : f)
        for (auto &phi : block.phis())
          if (phi.getType()->isAggregateType())
            aggregatePhis.push_back(&phi);
      for (auto *phi : aggregatePhis)
        DemotePHIToStack(phi, f.getEntryBlock().getFirstInsertionPt());
      SmallVector<AllocaInst *> allocas;
      for (auto &i : instructions(f))
        if (auto *a = dyn_cast<AllocaInst>(&i))
          allocas.push_back(a);
      for (auto *a : allocas) {
        if (!a->isStaticAlloca())
          return createStringError(inconvertibleErrorCode(),
                                   "OpenCL: only fixed entry-block private "
                                   "allocations are supported");
        if (a->getAddressSpace() == 5)
          continue;
        if (a->getAddressSpace() != 0)
          return createStringError(inconvertibleErrorCode(),
                                   "OpenCL: alloca must use private memory");
        IRBuilder<> b(a);
        auto *privateAlloca =
            b.CreateAlloca(a->getAllocatedType(), 5, a->getArraySize());
        privateAlloca->setAlignment(a->getAlign());
        auto *generic = b.CreateAddrSpaceCast(privateAlloca, a->getType());
        a->replaceAllUsesWith(generic);
        a->eraseFromParent();
      }
      SmallVector<FreezeInst *> freezes;
      for (auto &i : instructions(f))
        if (auto *freeze = dyn_cast<FreezeInst>(&i);
            freeze && freeze->getType()->isAggregateType())
          freezes.push_back(freeze);
      for (auto *freeze : freezes) {
        IRBuilder<> builder(freeze);
        auto *replacement =
            freezeLeaves(builder, freeze->getOperand(0), freeze->getType());
        freeze->replaceAllUsesWith(replacement);
        freeze->eraseFromParent();
      }
      LoopAnalysisManager lam;
      FunctionAnalysisManager am;
      CGSCCAnalysisManager cgam;
      ModuleAnalysisManager mam;
      PassBuilder pb;
      am.registerPass([] {
        return TargetIRAnalysis([](const Function &function) {
          return TargetTransformInfo(std::make_unique<OpenCLLegalizationTTI>(
              function.getParent()->getDataLayout()));
        });
      });
      pb.registerModuleAnalyses(mam);
      pb.registerCGSCCAnalyses(cgam);
      pb.registerFunctionAnalyses(am);
      pb.registerLoopAnalyses(lam);
      pb.crossRegisterProxies(lam, am, cgam, mam);
      FunctionPassManager passes;
      // OpenCL C has no masked memory operations. Legalize those operations
      // with generic TTI, preserving vector arithmetic and the shared
      // optimizer.
      passes.addPass(ScalarizeMaskedMemIntrinPass());
      passes.addPass(LegalizePointerVectorsPass());
      passes.addPass(InferAddressSpacesPass(0u));
      passes.addPass(SROAPass(SROAOptions::ModifyCFG));
      passes.addPass(InstCombinePass());
      passes.addPass(SimplifyCFGPass());
      passes.addPass(InstCombinePass());
      passes.addPass(LegalizePointerVectorsPass());
      passes.addPass(InstCombinePass());
      passes.run(f, am);
      // OpenCL C limits native vector values to 16 lanes. LLVM's shared
      // optimizer can produce wider values (notably in reduction monoids).
      // Legalize those functions with the standard scalarizer; leave native
      // SIMD kernels untouched. Run last so InstCombine cannot re-form wide
      // loads/stores after legalization.
      if (llvm::any_of(instructions(f), [](const Instruction &instruction) {
            auto *vector = dyn_cast<FixedVectorType>(instruction.getType());
            return vector && vector->getNumElements() > 16;
          })) {
        ScalarizerPassOptions options;
        options.ScalarizeLoadStore = true;
        ScalarizerPass(options).run(f, am);
      }
    }
    // Device failures are a backend transport concern, independent of the
    // logical argument ABI. Propagate the hidden error argument through the
    // reachable call graph, preserving ordinary Mojo assertion checks.
    for (auto &function : module)
      for (auto &instruction : instructions(function)) {
        auto *call = dyn_cast<CallBase>(&instruction);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        if (isa<UnreachableInst>(instruction) ||
            (callee && callee->getIntrinsicID() == Intrinsic::trap))
          errorFunctions.insert(&function);
        if (callee &&
            (callee->getName() == "__mojo_opencl_barrier" ||
             callee->getName() == "__mojo_opencl_subgroup_barrier_w64"))
          barrierFunctions.insert(&function);
      }
    bool changed;
    do {
      changed = false;
      for (auto &function : module)
        for (auto &instruction : instructions(function))
          if (auto *call = dyn_cast<CallBase>(&instruction))
            if (auto *callee = call->getCalledFunction()) {
              if (errorFunctions.contains(callee))
                changed |= errorFunctions.insert(&function).second;
              if (barrierFunctions.contains(callee))
                changed |= barrierFunctions.insert(&function).second;
            }
    } while (changed);
    for (auto &function : module)
      if (kernel(function) && errorFunctions.contains(&function) &&
          barrierFunctions.contains(&function))
        return createStringError(inconvertibleErrorCode(),
                                 "OpenCL: device assertions with workgroup "
                                 "barriers require safe workgroup termination");

    M::OpenCL::Bundle result;
    for (auto &global : module.globals()) {
      if (global.getAddressSpace() != 3 ||
          (global.hasInitializer() &&
           !isa<UndefValue>(global.getInitializer())))
        fail("OpenCL: only uninitialized shared globals are supported");
      names[&global] = "mojo_shared_" + std::to_string(next++);
    }
    for (auto &f : module)
      if (!f.isDeclaration()) {
        SmallPtrSet<const GlobalVariable *, 8> usedGlobals;
        auto visit = [&](auto &&self, const Value *value) -> void {
          if (auto *global = dyn_cast<GlobalVariable>(value))
            usedGlobals.insert(global);
          else if (auto *expression = dyn_cast<ConstantExpr>(value))
            for (const auto &operand : expression->operands())
              self(self, operand.get());
        };
        for (auto &block : f)
          for (auto &instruction : block)
            for (const auto &operand : instruction.operands())
              visit(visit, operand.get());
        for (auto &global : module.globals())
          if (usedGlobals.contains(&global))
            sharedGlobals[&f].push_back(&global);
        type(f.getReturnType());
        for (auto &arg : f.args())
          type(arg.getType());
        for (auto &block : f)
          for (auto &instruction : block) {
            type(instruction.getType());
            if (auto *atomic = dyn_cast<AtomicRMWInst>(&instruction)) {
              atomicOrder(atomic->getOrdering());
              usesInt64Atomics |= atomic->getType()->isIntegerTy(64);
            }
            if (auto *cas = dyn_cast<AtomicCmpXchgInst>(&instruction)) {
              atomicOrder(cas->getSuccessOrdering());
              atomicOrder(cas->getFailureOrdering());
              usesInt64Atomics |=
                  cas->getCompareOperand()->getType()->isIntegerTy(64);
            }
            if (auto *load = dyn_cast<LoadInst>(&instruction);
                load && load->isAtomic()) {
              atomicOrder(load->getOrdering());
              usesInt64Atomics |= load->getType()->isIntegerTy(64);
            }
            if (auto *store = dyn_cast<StoreInst>(&instruction);
                store && store->isAtomic()) {
              atomicOrder(store->getOrdering());
              usesInt64Atomics |=
                  store->getValueOperand()->getType()->isIntegerTy(64);
            }
            if (auto *fence = dyn_cast<FenceInst>(&instruction))
              atomicOrder(fence->getOrdering());
          }
        if (f.isVarArg())
          fail("OpenCL: variadic functions are unsupported");
        functions[&f] = kernel(f) ? "mojo_kernel_" + std::to_string(next++)
                                  : "mojo_f" + std::to_string(next++);
        for (auto &arg : f.args())
          names[&arg] = "v" + std::to_string(next++);
        for (auto &block : f) {
          labels[&block] = "b" + std::to_string(next++);
          for (auto &i : block)
            if (!i.getType()->isVoidTy())
              names[&i] = "v" + std::to_string(next++);
        }
        if (kernel(f)) {
          StringRef name(functions[&f]);
          if (name.empty() ||
              std::isdigit(static_cast<unsigned char>(name.front())))
            fail("OpenCL: invalid kernel identifier");
          for (char c : name)
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
              fail("OpenCL: invalid kernel identifier");
          if (!f.getReturnType()->isVoidTy())
            fail("OpenCL: kernel must return void");
          M::OpenCL::KernelABI abi{f.getName().str(), {}};
          abi.nativeName = functions[&f];
          abi.dynamicShared = dynamicShared(f);
          abi.deviceErrors = errorFunctions.contains(&f);
          SmallPtrSet<const Function *, 8> visited;
          abi.subgroupSize = usesSubgroups(f, visited) ? 64 : 0;
          for (auto &arg : f.args()) {
            auto *t = arg.getType();
            type(t);
            if (!error.empty())
              return createStringError(inconvertibleErrorCode(), "%s",
                                       error.c_str());
            if (arg.hasByValAttr() || arg.hasStructRetAttr())
              fail("OpenCL: unflattened aggregate kernel arguments are "
                   "unsupported");
            // Scalar i1 occupies one byte in the ABI and is emitted as uchar.
            // A boolean vector's packed LLVM layout does not match ucharN.
            if (t->isVectorTy() && t->getScalarType()->isIntegerTy(1))
              fail("OpenCL: packed boolean-vector kernel arguments are "
                   "unsupported");
            if (t->isPointerTy() && t->getPointerAddressSpace() != 0 &&
                t->getPointerAddressSpace() != 1)
              fail("OpenCL: kernel pointer arguments must use global memory");
          }
          if (auto it = aggregateABIs.find(&f); it != aggregateABIs.end())
            abi.arguments = it->second;
          result.kernels.push_back(std::move(abi));
        }
      }
    if (result.kernels.empty())
      fail("OpenCL: no exported kernel found");
    out << "// Generated by the Mojo OpenCL target.\n#pragma OPENCL "
           "FP_CONTRACT OFF\n";
    out << "typedef struct { uint old; uchar success; } mojo_cas32;\n"
           "typedef struct { ulong old; uchar success; } mojo_cas64;\n";
    if (usesInt64Atomics) {
      for (auto extension :
           {"cl_khr_int64_base_atomics", "cl_khr_int64_extended_atomics"}) {
        result.requiredExtensions.push_back(extension);
        out << "#pragma OPENCL EXTENSION " << extension << " : enable\n";
      }
    }
    if (usesAtomics) {
      result.openclCVersion = 300;
      out << "#ifndef __opencl_c_atomic_scope_device\n#error Device-scope "
             "atomics are required\n#endif\n";
      if (usesAcquireRelease)
        out << "#ifndef __opencl_c_atomic_order_acq_rel\n#error "
               "Acquire/release atomics are required\n#endif\n";
      if (usesSequentialConsistency)
        out << "#ifndef __opencl_c_atomic_order_seq_cst\n#error Sequentially "
               "consistent atomics are required\n#endif\n";
    }
    if (usesHalf) {
      result.requiredExtensions.push_back("cl_khr_fp16");
      out << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }
    if (module.getFunction("__mojo_opencl_shuffle_idx_u32_w64") ||
        module.getFunction("__mojo_opencl_subgroup_local_id") ||
        module.getFunction("__mojo_opencl_subgroup_barrier_w64")) {
      result.requiredExtensions.push_back("cl_khr_subgroups");
      out << "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
      if (module.getFunction("__mojo_opencl_shuffle_idx_u32_w64")) {
        result.requiredExtensions.push_back("cl_khr_subgroup_shuffle");
        out << "#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable\n";
      }
    }

    for (const auto &[element, name] : floatingPairs)
      out << "typedef struct { " << name << " first; " << name
          << " second; } mojo_pair_" << name << ";\n";

    for (const auto &[record, name] : numericRecords) {
      const auto &layout = module.getDataLayout();
      auto *offsets = layout.getStructLayout(record);
      out << "typedef struct __attribute__((packed,aligned("
          << layout.getABITypeAlign(record).value() << "))) {\n";
      uint64_t position = 0;
      for (unsigned field = 0; field < record->getNumElements(); ++field) {
        uint64_t offset = offsets->getElementOffset(field);
        if (offset > position)
          out << "uchar padding_" << field << "[" << offset - position
              << "];\n";
        out << type(record->getElementType(field)) << " "
            << fieldName(record, field) << ";\n";
        position =
            offset + layout.getTypeAllocSize(record->getElementType(field))
                         .getFixedValue();
      }
      uint64_t size = layout.getTypeAllocSize(record).getFixedValue();
      if (size > position)
        out << "uchar padding_end[" << size - position << "];\n";
      out << "} " << name << ";\n"
          << "typedef char " << name << "_size_check[sizeof(" << name
          << ")==" << size << "?1:-1];\n";
    }

    for (auto &f : module)
      if (!f.isDeclaration()) {
        signature(f);
        out << ";\n";
      }
    for (auto &f : module)
      if (!f.isDeclaration()) {
        currentFunction = &f;
        signature(f);
        out << " {\n";
        if (kernel(f))
          for (auto *global : sharedGlobals[&f]) {
            auto bytes = module.getDataLayout()
                             .getTypeAllocSize(global->getValueType())
                             .getFixedValue();
            if (!bytes)
              out << "__local uchar *" << names[global]
                  << " = mojo_dynamic_shared;\n";
            else
              out << "__local uchar " << names[global] << "[" << bytes
                  << "] __attribute__((aligned("
                  << global->getAlign().valueOrOne().value() << ")));\n";
          }
        if (kernel(f))
          for (auto &arg : f.args())
            if (arg.getType()->isPointerTy())
              out << type(arg.getType()) << " " << names[&arg] << " = "
                  << names[&arg] << "_base ? " << names[&arg] << "_base + "
                  << names[&arg] << "_offset : 0;\n";
        if (kernel(f) && errorFunctions.contains(&f))
          out << "if (atomic_or((volatile __global uint *)mojo_error, 0u)) "
                 "return;\n";
        for (auto &block : f)
          for (auto &i : block)
            if (!i.getType()->isVoidTy()) {
              if (auto *a = dyn_cast<AllocaInst>(&i)) {
                uint64_t count =
                    cast<ConstantInt>(a->getArraySize())->getZExtValue();
                uint64_t bytes = module.getDataLayout()
                                     .getTypeAllocSize(a->getAllocatedType())
                                     .getFixedValue();
                if (!count || !bytes || bytes > 1048576 ||
                    count > 1048576 / bytes)
                  fail("OpenCL: private allocation must be between 1 byte and "
                       "1 MiB");
                out << "__private uchar " << names[&i] << "_storage["
                    << count * bytes << "] __attribute__((aligned("
                    << a->getAlign().value() << ")));\n";
              }
              out << type(i.getType()) << " " << names[&i] << ";\n";
              if (isa<PHINode>(i))
                out << type(i.getType()) << " " << names[&i] << "_next;\n";
            }
        for (auto &block : f) {
          out << labels[&block] << ": ;\n";
          for (auto &i : block) {
            if (isa<PHINode>(i))
              continue;
            if (auto *insert = dyn_cast<InsertValueInst>(&i)) {
              auto *record = dyn_cast<StructType>(insert->getType());
              if (!record || insert->getNumIndices() != 1 ||
                  insert->getIndices()[0] >= record->getNumElements()) {
                fail("OpenCL: unsupported aggregate insertion");
                continue;
              }
              out << names[&i] << " = " << value(insert->getAggregateOperand())
                  << ";\n"
                  << names[&i] << "."
                  << fieldName(record, insert->getIndices()[0]) << " = "
                  << value(insert->getInsertedValueOperand()) << ";\n";
              continue;
            }
            auto *intrinsic = dyn_cast<IntrinsicInst>(&i);
            if (isa<UnreachableInst>(i) ||
                (intrinsic && intrinsic->getIntrinsicID() == Intrinsic::trap)) {
              out << "atomic_or((volatile __global uint *)mojo_error, 1u);\n";
              returnFromFunction(f);
              continue;
            }
            // Keep the shared optimizer's combined sine/cosine operation.
            // OpenCL sincos returns sine and writes cosine to private storage.
            if (isSinCos(&i)) {
              auto &call = cast<CallBase>(i);
              out << names[&i] << ".first = sincos("
                  << value(call.getArgOperand(0)) << ",&" << names[&i]
                  << ".second);\n";
              continue;
            }
            if (auto *memory = dyn_cast<MemIntrinsic>(&i)) {
              if (memory->isVolatile())
                fail("OpenCL: volatile memory intrinsic is unsupported");
              auto n = std::to_string(next++);
              auto dest = value(memory->getRawDest());
              auto length = value(memory->getLength());
              if (auto *set = dyn_cast<MemSetInst>(memory)) {
                out << "for (ulong m" << n << "=0; m" << n << "<" << length
                    << "; ++m" << n << ") " << dest << "[m" << n
                    << "] = " << value(set->getValue()) << ";\n";
              } else if (auto *copy = dyn_cast<MemTransferInst>(memory)) {
                auto src = value(copy->getRawSource());
                bool move = isa<MemMoveInst>(memory);
                if (move) {
                  if (copy->getRawDest()->getType()->getPointerAddressSpace() !=
                      copy->getRawSource()->getType()->getPointerAddressSpace())
                    fail("OpenCL: memmove requires matching address spaces");
                  out << "if (" << dest << " > " << src << ") { for (ulong m"
                      << n << "=" << length << "; m" << n << ">0; --m" << n
                      << ") " << dest << "[m" << n << "-1] = " << src << "[m"
                      << n << "-1]; } else ";
                }
                out << "{ for (ulong m" << n << "=0; m" << n << "<" << length
                    << "; ++m" << n << ") " << dest << "[m" << n
                    << "] = " << src << "[m" << n << "]; }\n";
              } else
                fail("OpenCL: unsupported memory intrinsic");
              continue;
            }
            if (auto *cas = dyn_cast<AtomicCmpXchgInst>(&i)) {
              auto *element = cas->getCompareOperand()->getType();
              if (cas->isVolatile() ||
                  cas->getAlign() <
                      module.getDataLayout().getABITypeAlign(element))
                fail("OpenCL: volatile or unaligned compare-exchange is "
                     "unsupported");
              out << names[&i] << ".old = " << value(cas->getCompareOperand())
                  << ";\n"
                  << names[&i] << ".success = atomic_compare_exchange_"
                  << (cas->isWeak() ? "weak" : "strong") << "_explicit("
                  << atomicPointer(cas->getPointerOperand(), element) << ",&"
                  << names[&i] << ".old," << value(cas->getNewValOperand())
                  << "," << atomicOrder(cas->getSuccessOrdering()) << ","
                  << atomicOrder(cas->getFailureOrdering()) << ","
                  << atomicScope(cas->getSyncScopeID()) << ");\n";
              continue;
            }
            if (auto *rmw = dyn_cast<AtomicRMWInst>(&i);
                rmw && rmw->getType()->isFloatTy()) {
              if (rmw->isVolatile() || rmw->getAlign() < Align(4))
                fail("OpenCL: volatile or unaligned floating atomic is "
                     "unsupported");
              std::string op;
              switch (rmw->getOperation()) {
              case AtomicRMWInst::FAdd:
                op = "+";
                break;
              case AtomicRMWInst::FSub:
                op = "-";
                break;
              case AtomicRMWInst::FMax:
                op = "fmax";
                break;
              case AtomicRMWInst::FMin:
                op = "fmin";
                break;
              case AtomicRMWInst::Xchg:
                break;
              default:
                fail("OpenCL: unsupported floating atomic operation");
                break;
              }
              auto n = names[&i];
              auto oldValue = "as_float(" + n + "_bits)";
              auto rhs = value(rmw->getValOperand());
              auto desired = op.empty() ? rhs
                             : (op == "fmin" || op == "fmax")
                                 ? op + "(" + oldValue + "," + rhs + ")"
                                 : oldValue + op + rhs;
              auto pointer =
                  atomicPointer(rmw->getPointerOperand(),
                                Type::getInt32Ty(module.getContext()));
              auto scope = atomicScope(rmw->getSyncScopeID());
              out << "{ uint " << n << "_bits = atomic_load_explicit("
                  << pointer << ",memory_order_relaxed," << scope << ");\n"
                  << "do { float " << n << "_desired = " << desired << ";\n"
                  << "if (atomic_compare_exchange_strong_explicit(" << pointer
                  << ",&" << n << "_bits,as_uint(" << n << "_desired),"
                  << atomicOrder(rmw->getOrdering()) << ",memory_order_relaxed,"
                  << scope << ")) break;\n"
                  << "} while (1); " << n << " = as_float(" << n
                  << "_bits); }\n";
              continue;
            }
            if (auto *insert = dyn_cast<InsertElementInst>(&i)) {
              out << names[&i] << " = " << value(insert->getOperand(0))
                  << ";\n";
              unsigned width =
                  cast<FixedVectorType>(i.getType())->getNumElements();
              for (unsigned lane = 0; lane < width; ++lane) {
                if (auto *index =
                        dyn_cast<ConstantInt>(insert->getOperand(2))) {
                  if (index->getZExtValue() != lane)
                    continue;
                } else
                  out << "if (" << value(insert->getOperand(2))
                      << " == " << lane << ") ";
                out << component(&i, lane) << " = "
                    << value(insert->getOperand(1)) << ";\n";
              }
              continue;
            }
            // LLVM 23 split BranchInst into two classes. The generic
            // instruction interface and textual opcode work with both forms.
            if (StringRef(i.getOpcodeName()) == "br") {
              if (i.getNumSuccessors() == 2) {
                out << "if (" << value(i.getOperand(0)) << ") {\n";
                edge(&block, i.getSuccessor(0));
                out << "} else {\n";
                edge(&block, i.getSuccessor(1));
                out << "}\n";
              } else
                edge(&block, i.getSuccessor(0));
            } else if (auto *dispatch = dyn_cast<SwitchInst>(&i)) {
              out << "switch (" << value(dispatch->getCondition()) << ") {\n";
              for (auto entry : dispatch->cases()) {
                out << "case " << value(entry.getCaseValue()) << ": {\n";
                edge(&block, entry.getCaseSuccessor());
                out << "}\n";
              }
              out << "default: {\n";
              edge(&block, dispatch->getDefaultDest());
              out << "}\n}\n";
            } else if (auto *ret = dyn_cast<ReturnInst>(&i)) {
              out << "return";
              if (ret->getReturnValue())
                out << " " << value(ret->getReturnValue());
              out << ";\n";
            } else if (auto *store = dyn_cast<StoreInst>(&i)) {
              if (store->getValueOperand()->getType()->isVectorTy() &&
                  store->getValueOperand()
                      ->getType()
                      ->getScalarType()
                      ->isIntegerTy(1))
                fail("OpenCL: packed boolean vector memory is unsupported");
              if (store->isAtomic()) {
                if (store->isVolatile() ||
                    store->getAlign() <
                        module.getDataLayout().getABITypeAlign(
                            store->getValueOperand()->getType()))
                  fail("OpenCL: volatile or unaligned atomic stores are "
                       "unsupported");
                bool floating =
                    store->getValueOperand()->getType()->isFloatTy();
                auto *element = floating ? Type::getInt32Ty(module.getContext())
                                         : store->getValueOperand()->getType();
                auto operand = value(store->getValueOperand());
                out << "atomic_store_explicit("
                    << atomicPointer(store->getPointerOperand(), element) << ","
                    << (floating ? "as_uint(" + operand + ")" : operand) << ","
                    << atomicOrder(store->getOrdering()) << ","
                    << atomicScope(store->getSyncScopeID()) << ");\n";
                continue;
              }
              if (store->isVolatile())
                fail("OpenCL: volatile stores are unsupported");
              if (store->getAlign() <
                  module.getDataLayout().getABITypeAlign(
                      store->getValueOperand()->getType()->getScalarType())) {
                unalignedStore(out, store->getValueOperand(),
                               store->getPointerOperand());
                continue;
              }
              if (auto *vector = dyn_cast<FixedVectorType>(
                      store->getValueOperand()->getType())) {
                if (store->getAlign().value() >=
                    nativeVectorAlignment(vector)) {
                  out << "*(("
                      << addressSpace(store->getPointerOperand()->getType())
                      << " " << type(vector) << "*)"
                      << value(store->getPointerOperand())
                      << ") = " << value(store->getValueOperand()) << ";\n";
                  continue;
                }
                out << "vstore" << vector->getNumElements() << "("
                    << value(store->getValueOperand()) << ", 0, ("
                    << addressSpace(store->getPointerOperand()->getType())
                    << " " << type(vector->getElementType()) << "*)"
                    << value(store->getPointerOperand()) << ");\n";
                continue;
              }
              out << "*(("
                  << memoryPointerType(store->getValueOperand()->getType(),
                                       store->getPointerOperand()->getType())
                  << ")" << value(store->getPointerOperand())
                  << ") = " << value(store->getValueOperand()) << ";\n";
            } else {
              if (!i.getType()->isVoidTy())
                out << names[&i] << " = ";
              out << expression(i) << ";\n";
              if (auto *call = dyn_cast<CallBase>(&i))
                if (auto *callee = call->getCalledFunction();
                    callee && !callee->isDeclaration() &&
                    errorFunctions.contains(callee)) {
                  out << "if (atomic_or((volatile __global uint *)mojo_error, "
                         "0u)) {\n";
                  returnFromFunction(f);
                  out << "}\n";
                }
            }
          }
        }
        out << "}\n";
      }
    if (!error.empty())
      return createStringError(inconvertibleErrorCode(), "%s", error.c_str());
    result.source = std::move(code);
    return result;
  }
};
} // namespace
Expected<M::OpenCL::Bundle> emitOpenCL(Module &module) {
  return Emitter(module).run();
}
} // namespace M::KGEN
