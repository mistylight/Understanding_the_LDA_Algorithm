import plotly
from plotly.colors import n_colors
import plotly.graph_objects as go

def visualize_topic_word(words, probs, max_prob=0.05):
    """
    words: List[List[str]], shape (num_topics, num_top_words)
        The top X words for each topic.
    probs: List[List[float]], shape (num_topics, num_top_words)
        Probabilities for the top X words for each topics.
    max_prob: float in (0, 1)
        The min probability to be shown as the deepest color.
    """
    LIGHT_GREEN = "rgb(238, 246, 237)"
    AVOCADO_GREEN = "rgb(55, 125, 34)"
    GREY = "rgb(220, 220, 220)"
    cmap = n_colors(LIGHT_GREEN, AVOCADO_GREEN, 10, colortype="rgb")
    
    header = [f"Topic {i:02d}" for i in range(10)]
    colors = [[cmap[int(min(p_ij / max_prob, 1) * 9)] for p_ij in p_i] for p_i in probs]

    fig = go.Figure(
        go.Table(
            header=dict(values=header, fill_color=GREY),
            cells=dict(values=words, fill_color=colors)
        )
    )
    fig.update_layout(
        title_text=f"Top words for the {len(words)} topics",
        width=1350,
    )
    fig.show("notebook")