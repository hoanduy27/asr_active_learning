import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as fin:
    lines = fin.readlines()
    lines = [o.strip() for o in lines]
    lines = [o for o in lines if len(o) > 0]
    req = [o for o in lines if not o.startswith('#') and not o.startswith('git+')]

setuptools.setup(

    name="asr_model_testing",  # Replace with your username

    version="0.0.1",

    author="Duy",

    author_email="hoanduy27@gmail.com",

    description="ASR Model Testing",

    long_description=long_description,

    install_requires=req,

    long_description_content_type="text/markdown",

    url="https://github.com/hoanduy27/asr_active_learning",

    packages=setuptools.find_packages(),

    python_requires='>=3.6',

)