<div align="center">
  
![](demo/temp_result.gif)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<div align="center">
  
*Данный репо является вспомогательным для [глобального проекта](https://github.com/Laggg/ml-bots-surviv.io) - создать ML-бота для игры [surviv.io](https://surviv.io/), за которым было бы интересно наблюдать.*
 
</div>

## Задача на ODS-хакатон 5.02-20.02
1. Игра surviv.io не имеет окружения, в котором можно было бы обучить RL-агента. Окружение, оно же environment - функция, принимающая в себя текущее состояние среды и действие агента, а возвращающая - следующее состояние и награду. Т.е. State_next,Reward=ENV(State_curr,Action).
2. Текущая версия агента получена с помощью алгоритмов offline reinforcement learning, которые не требуют окружения.
3. Идея - создать модель, выполняющую функции окружения, которая предсказывала бы следующий кадр, если известен текущий кадр и действие агента, т.е. State_next = MODEL(State_curr,Action). Награду отдельно можно захардкодить, это не проблема.
4. Идея не нова, существует статья/модель [GameGAN](https://nv-tlabs.github.io/gameGAN/) и [Dreamer2](https://youtu.be/o75ybZ-6Uu8?t=2).
5. Энтузиасты уже применили GameGAN к GTA, получив [GANTheftAuto](https://github.com/Sentdex/GANTheftAuto)
6. После создания модели - нейронного энвайрмента игры, можно обучить RL-агента уже с использованием окружения (научить лутать, правильно двигаться)
7. Данные есть, сервер для вычислений есть, идеи есть, а рук - не хватает, поэтому - присоединяйтесь
8. Сравнение моделей, которые есть в данный момент:

<div align="center">
  
![](demo/gif_dir6.gif)
  
</div>
