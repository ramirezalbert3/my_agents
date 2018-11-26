from setuptools import setup

setup(name='my_agents',
      version='0.1',
      description='Reinforcement learning agents, with Keras+TF built around openai-gym API',
      url='https://github.com/ramirezalbert3/my_agents',
      author='ramirezalbert3',
      license='MIT',
      packages=['my_agents', 'my_agents/agents', 'my_agents/core'],
      install_requires=[
          'gym', 'numpy', 'pandas', 'tensorflow', 'seaborn'
      ],
      zip_safe=False)
