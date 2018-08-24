import os
import time
import random

import argparse
import logging

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import init_cuda, prepare_data, tensors_from_pair, time_since, save_pickle
from network import EncoderRNN, AttnDecoderRNN
from evaluate import evaluate_randomly

plt.switch_backend('agg')

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 15

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Configuration')

    parser.add_argument('--epochs', type=int, default=200000, dest='epochs', 
                        help='Number of epochs for training')

    parser.add_argument('--lr', type=float, default=0.01, dest='lr',
                        help='Initial learning rate')

    parser.add_argument('--rnn-type', type=str, default='lstm', dest='rnn_type',
                        help='Type of the RNN layer')
    
    parser.add_argument('--num-layers', type=int, default=2, dest='num_layers',
                        help='Number of layers in the RNN models')

    parser.add_argument('--hidden-size', type=int, default=256, dest='hidden_size',
                        help='Number of hidden units in each layer of RNN')

    parser.add_argument('--bidirectional', type=bool, default=True, dest='bidirectional',
                        help='Bidirectional RNN')

    parser.add_argument('--dropout-rate', type=float, default=0.1, dest='dropout_p',
                        help='Dropout rate for the decoder RNN')
    
    parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5, dest='teacher_forcing_ratio',
                        help='Probability of teacher forcing the training of the model')
    
    parser.add_argument('--model-name', type=str, default='eng_deu.lstm_2_bi_sgd', dest='model_name',
                        help='Name for the model')

    parser.add_argument('--save-every', type=int, default=20000, dest='save_every',
                        help='Epoch interval after which to save the model')

    parser.add_argument('--print-every', type=int, default=1000, dest='print_every',
                        help='Epoch interval after which to print training state')

    parser.add_argument('--log-level', type=str, default='info', dest='log_level',
                        help='Logging level')
    
    return parser.parse_args()

def train(input_tensor, target_tensor, encoder, decoder, 
        encoder_optimizer, decoder_optimizer, loss_function, 
        teacher_forcing_ratio=0.5, max_length=MAX_LENGTH):
    
    encoder_hidden = encoder.init_hidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, 
        encoder.hidden_size, device=torch.device('cuda'))
    
    loss = 0
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]
        
    decoder_input = torch.tensor([[SOS_TOKEN]], device=torch.device('cuda'))
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += loss_function(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            
            loss += loss_function(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_TOKEN:
                break
                
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length

def train_epochs(encoder, decoder, pairs, input_lang, output_lang, epochs=10000, print_every=1000, 
                save_every=10000, plot_every=100, lr=0.01, teacher_forcing_ratio=0.5,
                lr_step_divisor=3, lr_step_gamma=0.1, model_name='autoencoder'):

    start = time.time()
    
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0 
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)

    encoder_lr_scheduler = StepLR(encoder_optimizer, 
        step_size=epochs // lr_step_divisor , gamma=lr_step_gamma)
    decoder_lr_scheduler = StepLR(decoder_optimizer, 
        step_size=epochs // lr_step_divisor , gamma=lr_step_gamma)
    
    training_pairs = [tensors_from_pair(random.choice(pairs), 
        input_lang, output_lang) for i in range(epochs)]
    
    loss_function = nn.NLLLoss()
    
    for epoch in range(1, epochs + 1):
        training_pair = training_pairs[epoch - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        
        loss = train(input_tensor, target_tensor, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, loss_function,
                     teacher_forcing_ratio=teacher_forcing_ratio)
        print_loss_total += loss
        plot_loss_total += loss
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Epoch {}/{}: {:.2f}% | Loss: {:.2f} | {}'
                 .format(epoch, epochs, (epoch / epochs) * 100,
                  print_loss_avg, time_since(start, epoch / epochs)))
        
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if epoch % save_every == 0:
            logging.info('Saving models on Epoch {}'.format(epoch))
            torch.save(encoder, 'models/{}/{}_{}.encoder' 
                .format(model_name, epoch, model_name))
            torch.save(decoder, 'models/{}/{}_{}.decoder' 
                .format(model_name, epoch, model_name))
        
    show_plot(plot_losses)

def show_plot(points):
    plt.figure()
    _, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def main():
    args = parse_arguments()

    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))

    init_cuda()

    input_lang, output_lang, train_pairs, test_pairs = \
        prepare_data('eng', 'deu', True)
    print(random.choice(train_pairs))

    if not os.path.isdir('models/{}'.format(args.model_name)):
        os.mkdir('models/{}'.format(args.model_name))

    logging.info('Saving input language, output language and testing data...')
    save_pickle((input_lang, output_lang, test_pairs), 'models/{}/{}.data'
        .format(args.model_name, args.model_name))

    encoder1 = EncoderRNN(input_lang.n_words, args.hidden_size, 
                bidirectional=args.bidirectional, layer_type=args.rnn_type, 
                num_layers=args.num_layers).cuda()
            
    attn_decoder1 = AttnDecoderRNN(args.hidden_size, output_lang.n_words, 
                    bidirectional=args.bidirectional, 
                    layer_type=args.rnn_type, num_layers=args.num_layers, 
                    dropout_p=args.dropout_p).cuda()

    logging.info('Training models...')
    train_epochs(encoder1, attn_decoder1, train_pairs, input_lang, output_lang, 
            epochs=args.epochs, print_every=args.print_every, save_every=args.save_every, 
            lr=args.lr, teacher_forcing_ratio=args.teacher_forcing_ratio,
            model_name=args.model_name)

    logging.info('Saving trained models...')
    torch.save(encoder1, 'models/{}/{}.encoder'
        .format(args.model_name, args.model_name))
    torch.save(attn_decoder1, 'models/{}/{}.decoder'
        .format(args.model_name, args.model_name))

    logging.info('Evaluating models...')
    evaluate_randomly(test_pairs, encoder1, attn_decoder1, input_lang, output_lang)
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')