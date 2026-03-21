from env.microgrid_env import MicrogridEnv

env = MicrogridEnv()
obs, _ = env.reset()
print('Env reset obs:', obs)

action = 0
obs2, reward, done, _, _ = env.step(action)
print('Step reward:', reward)