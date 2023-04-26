# gol-emulator
An attempt to build a ML-derived emulator for the game of life model.

# Data
We use a set of simulations conducted with the "Game of life" model (using my implementation in C++ here https://github.com/HuotPV/life-is-a-wheel).
More precisely, I made 1000 simulations of an 80x80 cells world initialized from random states and ran for 50 iterations. I prefered to make multiple short simulations rather than few long ones since the game of life quickly produces typical features and to have a large number of random samples in my data.
