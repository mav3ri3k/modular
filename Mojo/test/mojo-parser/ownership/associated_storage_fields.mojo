# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s

# An associated storage type must have the same flattened lifetime fields as
# field accesses on the specialized container. Counting the unresolved witness
# as a single field can underallocate the liveness bitvector.

trait StorageEngine:
    comptime Storage[T: TrivialRegisterPassable]: TrivialRegisterPassable


@fieldwise_init
struct Aggregate[T: TrivialRegisterPassable](TrivialRegisterPassable):
    var first: Self.T
    var second: Self.T
    var third: Self.T
    var fourth: Self.T


struct AggregateEngine(StorageEngine):
    comptime Storage[T: TrivialRegisterPassable]: TrivialRegisterPassable = Aggregate[T]


struct Container[Engine: StorageEngine, T: TrivialRegisterPassable](TrivialRegisterPassable):
    var storage: Self.Engine.Storage[Self.T]
    var tag: Self.T

    # CHECK-LABEL: lit.fn @"__init__(
    def __init__(out self: Container[AggregateEngine, Self.T], var storage: Aggregate[Self.T]) where Self.Engine == AggregateEngine:
        self.storage = storage
        self.tag = self.storage.fourth


# CHECK-LABEL: lit.fn @"read_storage(
def read_storage(value: Container[AggregateEngine, Int]) -> Int:
    return value.storage.fourth + value.tag
