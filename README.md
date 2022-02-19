<div align="center">
  
*This repo is support for [a global project](https://github.com/Laggg/ml-bots-surviv.io) - to create a smart ML bot for a browser-based multiplayer online game in the genre of battle royale [surviv.io](https://surviv.io/) which would be interesting to watch*  
  
</div>

<div align="center">
  
![](demo/prod_demo_game.gif)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<div align="center">
  
## Создать нейронный энвайрмент игры для обучения RL агента на основе реплеев - экспертных траекторий
  
</div>

1. Игра surviv.io не имеет окружения, в котором можно было бы обучить RL-агента. Окружение, оно же environment - функция, принимающая в себя текущее состояние среды и действие агента, а возвращающая - следующее состояние и награду. Т.е. State_next,Reward=ENV(State_curr,Action).
2. Текущая версия агента получена с помощью алгоритмов offline reinforcement learning, которые не требуют окружения.
3. Идея - создать модель, выполняющую функции окружения, которая предсказывала бы следующий кадр, если известен текущий кадр и действие агента, т.е. State_next = MODEL(State_curr,Action). Награду отдельно можно захардкодить, это не проблема.
4. Идея не нова, существует статья/модель [GameGAN](https://nv-tlabs.github.io/gameGAN/) и [Dreamer2](https://youtu.be/o75ybZ-6Uu8?t=2).
5. Энтузиасты уже применили GameGAN к GTA, получив [GANTheftAuto](https://github.com/Sentdex/GANTheftAuto)
6. После создания модели - нейронного энвайрмента игры, можно обучить RL-агента уже с использованием окружения (научить лутать, правильно двигаться)
7. Сравнение генеративных моделей **S_next=model(S_curr,action)** (слева направо):
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
- [x] 1. генеративные модели **без использования GAN** для создания нейронного энвайрмента (см ./demo/gif_dir6.gif и ./demo/gif_dir3.gif)
- [x] 2. pix2pix
- [x] 3. VQ-VAE
- [ ] 4. GameGan/Dreamer2
- [x] 5. доп. [лоссы](https://www.youtube.com/watch?v=nUjIG41M8fM): /mssim/perceptual path length/texture loss/style loss/ - применимо ли к нашему домену?
- [x] 6. **RL** для приближения агента к кустам/камням/луту (в зависимости от качества нейронного движка)
- [x] 7. интерактивный фронт для взаимодействия человека с нейронным движком (чтобы можно было поиграть игру, движком которой была бы нейронка)
