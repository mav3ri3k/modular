from test_graph_storage_engine import main as test_main


@export("run_opencl_device_context")
def check_device_context() abi("C") -> Int32:
    try:
        test_main()
        print(
            "PASS: GraphStorageEngine opaque host tiles, offsets and launches"
        )
        return 0
    except e:
        print("FAIL:", e)
        return 1
