import time
import random

import torch
import torch.nn as nn

from torch import optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import get_torch_device, prepare_data, tensors_from_pair, time_since, save_pickle
from network import EncoderRNN, AttnDecoderRNN
from evaluate import evaluate_randomly

plt.switch_backend('agg')

SOS_TOKEN = 0
EOS_TOKEN = 1

MAX_LENGTH = 10

TEACHER_FORCING_RATIO = 0.5

HIDDEN_SIZE = 256

def train(input_tensor, target_tensor, encoder, decoder, 
        encoder_optimizer, decoder_optimizer, loss_function, max_length=MAX_LENGTH):
    
    encoder_hidden = encoder.init_hidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, 
                        encoder.hidden_size, device=device)
    
    loss = 0
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += loss_function(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach() # detach from history as input
            
            loss += loss_function(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_TOKEN:
                break
                
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length

def train_epochs(encoder, decoder, pairs, epochs=10000, print_every=1000, 
        save_every=10000, plot_every=100, learning_rate=0.001):

    start = time.time()
    
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0 
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    training_pairs = [tensors_from_pair(random.choice(pairs), input_lang, output_lang) for i in range(epochs)]
    
    # using the F.log_softmax() in decoder 
    # so no need for cross entropy loss
    loss_function = nn.NLLLoss()
    
    for epoch in range(1, epochs + 1):
        training_pair = training_pairs[epoch - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        loss = train(input_tensor, target_tensor, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, loss_function)
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
            debug('Saving models on Epoch {}'.format(epoch))
            torch.save(encoder, 'models/encoder1.fra_eng_{}'.format(epoch))
            torch.save(decoder, 'models/attn_decoder1.fra_eng_{}'.format(epoch))
        
    show_plot(plot_losses)

def show_plot(points):
    plt.figure()
    _, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def debug(message):
    print('\nDEBUG: {}\n'.format(message))

    
if __name__ == '__main__':

    '''Using CUDA Device'''
    device = get_torch_device()
    device_idx = torch.cuda.current_device()
    device_cap = torch.cuda.get_device_capability(device_idx)
    debug('PyTorch using {} device {}:{} with Compute Capability {}.{}'
        .format(str(device).upper(), torch.cuda.get_device_name(device_idx), device_idx, device_cap[0], device_cap[1]))

    input_lang, output_lang, train_pairs, test_pairs = prepare_data('eng', 'fra', True)
    print(random.choice(train_pairs))

    debug('Saving training and testing data...')
    save_pickle((train_pairs, test_pairs), 'data/eng-fra.data')

    encoder1 = EncoderRNN(input_lang.n_words, HIDDEN_SIZE, layer_type='lstm', num_layers=2).to(device)
    attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, layer_type='lstm', num_layers=2, dropout_p=0.3).to(device)

    debug('Training models...')

    train_epochs(encoder1, attn_decoder1, train_pairs, epochs=100000)

    debug('Saving trained models...')
    torch.save(encoder1, 'models/encoder1.fra_eng.model')
    torch.save(attn_decoder1, 'models/attn_decoder1.fra_eng.model')

    debug('Evaluating models...')
    evaluate_randomly(test_pairs, encoder1, attn_decoder1, input_lang)