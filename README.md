<div align="center">
  
![](demo/prod_demo_game.gif)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<div align="center">
  
*This repo is support for [a global project](https://github.com/Laggg/ml-bots-surviv.io) - to create a smart ML bot for a browser-based multiplayer online game in the genre of battle royale [surviv.io](https://surviv.io/) which would be interesting to watch*  
  
</div>

## Neural environment for training an RL-agent by using expert trajectories only

[surviv.io](https://surviv.io/) does not have an environment in which it would be possible to train an RL agent. Environment is a function that gets the current state and the agent's action and returns the next state and reward: **state_next,reward=ENV(state_curr,action)**. That's why we present our neural environment, trained with  100 youtube-videos containing 1.2 million frames (equivalent to ~12 hours of gameplay recordings). Our DL models perform all required functions of the real environment features needed for training RL-agents.

2. Идея - создать модель, выполняющую функции окружения, которая предсказывала бы следующий кадр, если известен текущий кадр и действие агента, т.е. State_next = MODEL(State_curr,Action). Награду отдельно можно захардкодить, это не проблема.
3. Сравнение генеративных моделей **S_next=model(S_curr,action)** (слева направо):
    - кадр старта
    - Loss = 0\*Lgan + MSE
    - Loss = 0\*Lgan + MSE + PL/100
    - Loss = 0\*Lgan + MAE + PL/100
    - Loss = 0\*Lgan + 3\*MAE + PL/100 (пока самая лучшая)
    - Loss = Lgan/100 + 3\*MAE + PL/100 (pix2pix)
 
<div align="center">
 
![](demo/gif_dir3.gif)
![](demo/gif_dir6.gif)

  
</div>

#### План работ:
- [x] 1. generative models without GAN
- [x] 2. generative models with GAN (pix2pix)
- [x] 3. VQ-VAE/[GameGAN](https://nv-tlabs.github.io/gameGAN/)/[Dreamer2](https://youtu.be/o75ybZ-6Uu8?t=2)
- [x] 4. additional [losses](https://www.youtube.com/watch?v=nUjIG41M8fM), /mssim/style-texture loss/perceptual path length/
- [x] 5. **RL** для приближения агента к кустам/камням/луту (в зависимости от качества нейронного движка)
- [x] 6. интерактивный фронт для взаимодействия человека с нейронным движком (чтобы можно было поиграть игру, движком которой была бы нейронка)
