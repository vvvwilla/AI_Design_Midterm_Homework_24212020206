# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-business-char'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'business-char'
wandb_run_name = 'mini-gpt'

dataset = 'business_char'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.5

learning_rate = 5e-4 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
