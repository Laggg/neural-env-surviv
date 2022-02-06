<div align="center">
  
![](jupyter_demo/temp_result.gif)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Slack](./jupyter_demo/slack.svg)](https://opendatascience.slack.com/archives/CJW0A6U78/p1632648992121300?thread_ts=1632648992.121300&cid=CJW0A6U78)

[![os](https://img.shields.io/badge/Linux-passing-success)]()
[![os](https://img.shields.io/badge/MacOS-passing-success)]()
[![os](https://img.shields.io/badge/Windows-passing-success)]()
</div>

## Задача на ODS-хакатон 5.02-20.02
1. Игра surviv.io не имеет окружения, в котором можно было бы обучить RL-агента. Окружение, оно же environment - функция, принимающая в себя текущее состояние среды и действие агента, а возвращающая - следующее состояние и награду. Т.е. State_next,Reward=ENV(State_curr,Action).
2. Текущая версия агента получена с помощью алгоритмов offline reinforcement learning, которые не требуют окружения.
3. Идея - создать модель, выполняющую функции окружения, которая предсказывала бы следующий кадр, если известен текущий кадр и действие агента, т.е. State_next = MODEL(State_curr,Action). Награду отдельно можно захардкодить, это не проблема.
4. Идея не нова, существует статья/модель [GameGAN](https://nv-tlabs.github.io/gameGAN/) и [Dreamer2](https://youtu.be/o75ybZ-6Uu8?t=2).
5. Энтузиасты уже применили GameGAN к GTA, получив [GANTheftAuto](https://github.com/Sentdex/GANTheftAuto)
6. После создания модели - нейронного энвайрмента игры, можно обучить RL-агента уже с использованием окружения (научить лутать, правильно двигаться)
7. Данные есть, сервер для вычислений есть, идеи есть, а рук - не хватает, поэтому - присоединяйтесь
8. Уже есть некоторые наработки (без применения GameGAN) по нейронному движку (нужно улучшать). Ниже пример одной из моделей, которая возвращает следующий кадр агенту: слева - начальный кадр, справа - некоторая траектория рандомного агента (который случайным образом выбирает направление своего движения)
<div align="center">
  
![](jupyter_demo/neural_engine.gif)
  
</div>

9. Сравнение моделей, которые есть в данный момент:
<div align="center">
  
![](jupyter_demo/gif_dir6.gif)
  
</div>
