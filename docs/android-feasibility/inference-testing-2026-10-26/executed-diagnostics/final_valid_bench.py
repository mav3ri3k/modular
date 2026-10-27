import pathlib,subprocess,shlex,json
out=pathlib.Path('/Users/REDACTED/projects/modular/docs/android-feasibility/inference-testing-2026-10-26');adb='/Users/REDACTED/Library/Android/sdk/platform-tools/adb';base='/data/local/tmp/modular-inference-testing'
for name,file in [('bonsai','Ternary-Bonsai-2-2B-PTQ1_0.gguf'),('qwen3','Qwen_Qwen3-1.7B-Q4_K_M.gguf')]:
 args=['env','GGML_VK_DISABLE_ASYNC=1','GGML_VK_DISABLE_FUSION=1','./llama-bench','-m','../'+file,'-p','0','-n','128','-r','3','-t','4','-ngl','99','-b','1','-ub','1','-o','json']
 result=subprocess.run([adb,'shell','cd '+base+'/prism-vulkan && LD_LIBRARY_PATH=. '+shlex.join(args)],check=True,capture_output=True,text=True)
 (out/(name+'-qualified-vulkan-tg128.stdout')).write_text(result.stdout);(out/(name+'-qualified-vulkan-tg128.stderr')).write_text(result.stderr)
 print(name,json.loads(result.stdout)[0]['avg_ts'],flush=True)
(out/'valid-reference-bench-complete.json').write_text(json.dumps({'complete':True}))
