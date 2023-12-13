# NEURAL NETWORK (in progress)
#### README is still in progress
### Build
Install [SFML](https://www.sfml-dev.org/)
```
sudo apt-get install libsfml-dev
```
Build library and demos:
```
mkdir build
cd build
cmake ..
cd ..
cmake --build build
```

### Run demos
Xor:
```
build/bin/xor
```

Digit recognition:
Training model:
```
build/bin/train {model number} {num of first iteration} {max iteration} {0/1; 1-load model, 0-no}
```
Test model:
```
build/bin/test {path to model}
```

Run an application for drawing numbers and recognizing them:
```
build/bin/visual_app
```
