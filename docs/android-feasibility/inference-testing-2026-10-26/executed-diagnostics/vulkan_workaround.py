import pathlib,subprocess,json,numpy as np
out=pathlib.Path('/Users/REDACTED/projects/modular/docs/android-feasibility/inference-testing-2026-10-26');b=pathlib.Path('/tmp/modular-inference-testing');adb='/Users/REDACTED/Library/Android/sdk/platform-tools/adb';base='/data/local/tmp/modular-inference-testing';records={}
for name,env in [('no-async-fusion','GGML_VK_DISABLE_ASYNC=1 GGML_VK_DISABLE_FUSION=1'),('no-f16','GGML_VK_DISABLE_F16=1')]:
 cmd=f'cd {base}/prism-vulkan; LD_LIBRARY_PATH=. {env} ./prism-reference-configurable ../Qwen_Qwen3-1.7B-Q4_K_M.gguf ../qwen3-tokens.txt ../vulkan-{name}.f32 99 f16 128'
 res=subprocess.run([adb,'shell',cmd],text=True,capture_output=True,check=True);(out/('vulkan-'+name+'.log')).write_text(res.stdout+'\n'+res.stderr)
 subprocess.run([adb,'pull',base+'/vulkan-'+name+'.f32',str(b)],check=True)
 x=np.fromfile(b/'qwen3-torch.f32',dtype='<f4').reshape(-1,151936).astype(np.float64);y=np.fromfile(b/('vulkan-'+name+'.f32'),dtype='<f4').reshape(x.shape).astype(np.float64)
 d=dict(max_abs=float(np.abs(x-y).max()),relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),argmax_agreement=float((x.argmax(1)==y.argmax(1)).mean()),finite=bool(np.isfinite(y).all()));records[name]=d;(out/'vulkan-workarounds.json').write_text(json.dumps(records,indent=2)+'\n');print(name,d,flush=True)
