# vallex.cpp
This is a port of [Plachtaa/VALL-E-X](https://github.com/Plachtaa/VALL-E-X) to [ggerganov/ggml](https://github.com/ggerganov/ggml). 

## Build

```commandline
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 
cmake --build . --config Release
```