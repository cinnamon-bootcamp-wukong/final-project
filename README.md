# final-project


## Use with Docker
- Install the NVIDIA Container Toolkit
- Run `docker build . -t wukong-avatar-creator`
- Run `docker run --gpus=all -p 3000:3000 wukong-avatar-creator`
- Check the web app at `https://localhost:3000`