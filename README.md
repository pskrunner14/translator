# Overview

*Translator* is a Machine Translation Autoencoder built using Seq2Seq Recurrent Nets with Attention in PyTorch.

## Dataset

You can download compiled and selected translated sentence pairs from [ManyThings.org](http://www.manythings.org/anki/) or directly from [Tatoeba](https://tatoeba.org/eng) into `data/` directory for training and testing.

## Getting Started

In order to train the model and start translating, you'll need to install the required python packages:

```
pip install -r requirements.txt
```

Now you can open up a terminal and start training the model, tweak your model configuration however you like:

```
python train.py --epochs 100000 --lr 0.001 --rnn-type lstm --num-layers 2 --hidden-size 256
```

Once you're done training, run the evaluation script which will load the pretrained models you provide in the `models/[model_name]` directory:

```
python evaluate.py --num-tests 30 --model-name lstm_2_bi_sgd
```

Now you are all set up!

## Results

Translations from French to English

```
Input: elle est habituee a vivre seule .
Target: she is used to living alone .
Predicted: she is used to living alone . <EOS>

Input: je suis sur que tes intentions etaient bonnes .
Target: i m sure your intentions were good .
Predicted: i am sure your intentions were good . <EOS>

Input: nous sommes attaquees .
Target: we re under attack .
Predicted: we re starved . <EOS>

Input: je ne suis pas laide .
Target: i m not ugly .
Predicted: i m not angry . <EOS>

Input: nous sommes juste fatigues .
Target: we re just tired .
Predicted: we re just bored . <EOS>    

Input: nous nous rapprochons .
Target: we re getting closer .
Predicted: we re deluding . <EOS>

Input: elle n a pas peur des serpents .
Target: she isn t afraid of snakes .
Predicted: she is not afraid of snakes . <EOS>

Input: c est deja un homme .
Target: he s already a man . 
Predicted: he is a man . <EOS>

Input: je me rejouis de vous avoir finalement rencontre .
Target: i m glad to finally meet you .
Predicted: i m glad to finally meet you . <EOS>

Input: je suis la pour te proteger .
Target: i m here to protect you .
Predicted: i m here to protect you . <EOS>
                                                                                                   
Input: il est toujours en train de se plaindre .
Target: he is constantly complaining .
Predicted: he is always dreaming . <EOS>

Input: ce ne sont pas mes regles .
Target: they re not my rules .
Predicted: they re not my . <EOS>                                                                  
                                                                                                   
Input: je suis desireux de t aider .                                                               
Target: i am willing to help you .                                                                 
Predicted: i am willing to help you . <EOS>                                                        
                                                                                                   
Input: je suis content de ne pas etre tom .                                                        
Target: i m glad i m not tom .                                                                     
Predicted: i m happy i not tom . <EOS>                                                
                                                                                                           
Input: vous etes la doyenne .                                                                      
Target: you re the oldest .                                                                        
Predicted: you re the oldest . <EOS>                                                               
                                                                                                   
Input: vous etes trop jeunes pour voyager seuls .                                                  
Target: you are too young to travel alone .                                                        
Predicted: you re too young to travel alone . <EOS>                                                
                                                                                                   
Input: je suis tres desorientee .                                                                  
Target: i m all confused .                                                                         
Predicted: i m very lonely . <EOS>                                                                 
                                                                                                   
Input: je suis ton jules n est ce pas ?                                                            
Target: i m your boyfriend aren t i ?                                                              
Predicted: i m the one aren t i ? <EOS>
```

## References

* [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)

## Built With

* Python
* PyTorch
* TorchText

## Authors

Prabhsimran Singh

## License

This project is licensed under the MIT License - see the LICENSE.md file for details