import pathlib,subprocess,json,numpy as np
r=pathlib.Path('/Users/REDACTED/projects/modular');out=r/'docs/android-feasibility/inference-testing-2026-10-26';b=pathlib.Path('/tmp/modular-inference-testing');adb='/Users/REDACTED/Library/Android/sdk/platform-tools/adb';base='/data/local/tmp/modular-inference-testing'
items=[('bonsai-pq','Ternary-Bonsai-2-2B-PQ2_0.gguf','bonsai-parity-tokens.txt','bonsai-parity-bonsai.f32',248320),('bonsai-ptq','Ternary-Bonsai-2-2B-PTQ1_0.gguf','bonsai-parity-tokens.txt','bonsai-parity-bonsai.f32',248320),('qwen3','Qwen_Qwen3-1.7B-Q4_K_M.gguf','qwen3-tokens.txt','qwen3-torch.f32',151936),('qwen35','Qwen_Qwen3.5-2B-Q4_K_M.gguf','qwen35-tokens.txt','qwen35-torch.f32',248320)]
records={}
for name,file,tokens,ref,vocab in items:
 cmd=f'cd {base}/prism-vulkan; LD_LIBRARY_PATH=. ./prism-reference-configurable ../{file} ../{tokens} ../{name}-vulkan-f16.f32 99 f16 128'
 res=subprocess.run([adb,'shell',cmd],text=True,capture_output=True,check=True);(out/(name+'-vulkan-f16.log')).write_text(res.stdout+'\n'+res.stderr)
 subprocess.run([adb,'pull',base+'/'+name+'-vulkan-f16.f32',str(b)],check=True)
 x=np.fromfile(b/ref,dtype='<f4').reshape(-1,vocab).astype(np.float64);y=np.fromfile(b/(name+'-vulkan-f16.f32'),dtype='<f4').reshape(x.shape).astype(np.float64)
 d=dict(max_abs=float(np.abs(x-y).max()),relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),argmax_agreement=float((x.argmax(1)==y.argmax(1)).mean()),reference_ids=x.argmax(1).tolist(),actual_ids=y.argmax(1).tolist(),finite=bool(np.isfinite(y).all()))
 records[name]=d;(out/'native-vulkan-f16-parity.json').write_text(json.dumps(records,indent=2)+'\n');print(name,d['max_abs'],d['relative_l2'],d['argmax_agreement'],flush=True)
(out/'native-default-checks-complete.json').write_text(json.dumps({'complete':True}))
