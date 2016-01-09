

def visualize_similarities_text(final_embeddings_similarity,reverse_dictionary,valid_examples):
  for i in xrange(valid_examples.size):
      valid_word = reverse_dictionary[valid_examples[i]]
      top_k = 8 # number of nearest neighbors
      nearest = (-final_embeddings_similarity[i, :]).argsort()[1:top_k+1]
      log_str = "Nearest to %s:" % valid_word
      for k in xrange(top_k):
        close_word = reverse_dictionary[nearest[k]]
        log_str = "%s %s," % (log_str, close_word)
      print(log_str)



def visualize_plot_with_labels(final_embeddings, reverse_dictionary, filename='tsne.png'):
  try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]

    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
      x, y = low_dim_embs[i,:]
      plt.scatter(x, y)
      plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)

  except ImportError:
     print("Please install sklearn and matplotlib to visualize embeddings.")
