import compiler_gym

def main():
    print("1. Initializing Environment...")
    # Create the environment WITHOUT the benchmark first
    with compiler_gym.make("llvm-v0") as env:
        
        # Explicitly ensure the dataset is ready
        if "cBench-v1" not in env.datasets:
            print("   Dataset missing. Downloading cBench-v1...")
            env.datasets["cBench-v1"].install()
        
        print("2. Loading Cartridge (cBench-v1/crc32)...")
        # Load the benchmark here
        env.reset(benchmark="cBench-v1/crc32")
        
        print("3. Extracting 'Autophase' Features...")
        # Extract the observation manually
        observation = env.observation["Autophase"]
        
        print(f"\n--- SUCCESS ---")
        print(f"First 10 Features: {observation[:10]}")
        print(f"Total Features: {len(observation)}")

if __name__ == "__main__":
    main()