import setuptools

setuptools.setup(
    name="speechEmotion",
    version="0.0.1",
    author="Antek",
    description="End-to-End Speech Emotion Recognition MLOps",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)