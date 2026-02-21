import compiler_gym
import gym

print("1. Core Library Imported.")

# The critical test: Does the Environment load without crashing?
try:
    env = compiler_gym.make("llvm-autophase-ic-v0")
    print("2. Compiler Environment Created Successfully.")
    
    env.reset(benchmark="cBench-v1/crc32")
    print("3. Benchmark (CRC32) Loaded.")
    
    action = env.action_space.sample()
    print(f"4. Testing Action: {action}")
    
    observation, reward, done, info = env.step(action)
    print(f"5. Action Executed. Reward: {reward}")
    
    env.close()
    print("\n[SUCCESS] The Agentic Compiler Engine is OPERATIONAL.")
except Exception as e:
    print(f"\n[FAIL] Engine Crashed: {e}")