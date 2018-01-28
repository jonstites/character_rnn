from setuptools import setup

setup(name='character_rnn',
      version='0.1',
      description='A character-by-character RNN.',
      url='http://github.com/jonstites/character_rnn',
      author='Jonathan Stites',
      author_email='mail@jonstites.com',
      license='MIT',
      packages=['character_rnn'],
      install_requires=[
          'argh'
          ],
      zip_safe=False)
