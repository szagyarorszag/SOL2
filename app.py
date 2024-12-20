import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import community as community_louvain
import uuid
import plotly.express as px

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Preprocesses the input text by tokenizing, lowercasing, removing stopwords,
    removing non-alphabetic tokens, and lemmatizing.
    """
    if not isinstance(text, str):
        return ""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


@st.cache_data
def load_data(csv_path):
    """
    Loads the dataset from the specified CSV path.
    Expects columns: arxiv_id, title, authors, abstract, categories,
    submission_date, last_revision_date, pdf_url, abs_url
    """
    column_names = [
        'arxiv_id',
        'title',
        'authors',
        'abstract',
        'categories',
        'submission_date',
        'last_revision_date',
        'pdf_url',
        'abs_url'
    ]
    try:
        df = pd.read_csv(
            csv_path,
            names=column_names,
            header=0,
            quotechar='"',
            escapechar='\\',
            engine='python',
            on_bad_lines='skip'
        )
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_resource
def create_graph(df, similarity_threshold=0.2):
    """
    Creates a NetworkX graph from the DataFrame.
    Nodes represent papers, and edges represent semantic similarity
    between abstracts above the specified threshold.
    Performs community detection using the Louvain method.
    """
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['arxiv_id'],
                   title=row['title'],
                   authors=row['authors'],
                   abstract=row['abstract'])

    df['processed_abstract'] = df['abstract'].apply(preprocess_text)

    original_len = len(df)
    df = df[df['processed_abstract'] != ""].reset_index(drop=True)
    filtered_len = len(df)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_abstract'])

    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            similarity = cosine_sim_matrix[i][j]
            if similarity >= similarity_threshold:
                paper_i = df.iloc[i]['arxiv_id']
                paper_j = df.iloc[j]['arxiv_id']
                G.add_edge(paper_i, paper_j, weight=similarity)
    partition = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition, 'community')

    return G, tfidf_vectorizer, tfidf_matrix, partition


def embed_user_article(title, abstract, tfidf_vectorizer):
    """
    Embeds the user's article by preprocessing and vectorizing the input.
    Returns the TF-IDF vector or None if the processed text is empty.
    """
    combined_text = f"{title} {abstract}"
    processed_text = preprocess_text(combined_text)
    if processed_text == "":
        return None
    vector = tfidf_vectorizer.transform([processed_text])
    return vector


def assign_community(user_vector, tfidf_matrix, df, partition):
    """
    Assigns the user's article to a community based on the highest similarity
    with existing papers.
    Returns the community ID, most similar paper ID, and similarity score.
    """
    if user_vector is None:
        return -1, None, 0.0
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    most_similar_idx = similarities.argmax()
    most_similar_paper_id = df.iloc[most_similar_idx]['arxiv_id']
    user_community = partition.get(most_similar_paper_id, -1)
    similarity_score = similarities[most_similar_idx]
    return user_community, most_similar_paper_id, similarity_score


def get_community_papers(df, community_id, G):
    """
    Retrieves all papers within the specified community.
    """
    community_papers = [node for node, data in G.nodes(data=True) if data.get('community') == community_id]
    return df[df['arxiv_id'].isin(community_papers)]


def plot_graph_plotly(G, partition, user_nodes=None, graph_type='full'):
    """
    Plots the NetworkX graph using Plotly.
    Optionally highlights user-submitted nodes with different shapes and sizes
    based on the graph type ('full' or 'community').
    """
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    communities = set(partition.values())
    num_communities = len(communities)
    color_palette = px.colors.qualitative.Alphabet if num_communities <= 26 else px.colors.qualitative.G10
    colors = {comm: color_palette[i % len(color_palette)] for i, comm in enumerate(communities)}

    node_x = []
    node_y = []
    node_color = []
    node_text = []
    node_size = []
    node_symbol = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        community = partition[node]
        node_color.append(colors[community])
        node_text.append(G.nodes[node]['title'])

        if user_nodes and node in user_nodes:
            if graph_type == 'community':
                symbol = 'square'  # User nodes as squares in community graph
                size = 10  # Standard size
            elif graph_type == 'updated':
                symbol = 'circle'  # User nodes as circles in updated graph
                size = 20  # Twice the standard size
            else:
                symbol = 'diamond'  # Default symbol
                size = 15  # Default size
        else:
            symbol = 'circle'  # Regular nodes as circles
            size = 10  # Standard size

        node_size.append(size)
        node_symbol.append(symbol)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2)
        ),
        text=node_text,
        hoverinfo='text',
        marker_symbol=node_symbol
    )

    data = [edge_trace, node_trace]

    fig = go.Figure(data=data,
                    layout=go.Layout(
                        title='🌐 Semantic Citation Map',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


def main():
    """
    The main function that runs the Streamlit app.
    """
    st.set_page_config(page_title="Semantic Citation Map Explorer", layout="wide")
    st.title("📚 Semantic Citation Map Explorer")

    st.sidebar.header("🔧 Configuration")

    csv_path = st.sidebar.text_input("📂 CSV File Path", "arxiv_large_diverse_dataset.csv")

    df = load_data(csv_path)

    if df is not None:
        st.success("✅ Dataset loaded successfully!")
        st.write(f"**Total Papers:** {len(df)}")
        if 'user_nodes' not in st.session_state:
            st.session_state.user_nodes = []
        if 'G' not in st.session_state:
            with st.spinner('🔨 Creating graph...'):
                G, tfidf_vectorizer, tfidf_matrix, partition = create_graph(df)
                st.session_state.G = G
                st.session_state.tfidf_vectorizer = tfidf_vectorizer
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.partition = partition
                st.success("🔗 Graph created successfully!")
        else:
            G = st.session_state.G
            tfidf_vectorizer = st.session_state.tfidf_vectorizer
            tfidf_matrix = st.session_state.tfidf_matrix
            partition = st.session_state.partition
            user_nodes = st.session_state.user_nodes

        st.header("🌐 Citation Network Visualization")
        if st.button("🚀 Generate Interactive Graph", key="generate_graph"):
            with st.spinner('📊 Generating graph...'):
                fig = plot_graph_plotly(G, partition, user_nodes=st.session_state.user_nodes, graph_type='full')
                st.plotly_chart(fig, use_container_width=True)

        st.header("✍️ Add Your Article to the Citation Map")
        with st.form("user_article_form"):
            user_title = st.text_input("📝 Article Title")
            user_abstract = st.text_area("📄 Article Abstract")
            submit_button = st.form_submit_button(label="➕ Submit")

        if submit_button:
            if user_title.strip() == "" or user_abstract.strip() == "":
                st.warning("⚠️ Please provide both the title and abstract of your article.")
            else:
                user_vector = embed_user_article(user_title, user_abstract, tfidf_vectorizer)

                user_community, similar_paper_id, similarity_score = assign_community(user_vector, tfidf_matrix, df,
                                                                                      partition)

                if user_community == -1:
                    st.error(
                        "❌ Could not assign the article to any community. Please ensure the abstract contains meaningful content.")
                else:
                    st.success(f"✅ Your article has been assigned to **Community {user_community}**")
                    st.write(f"**Most Similar Paper ID:** {similar_paper_id}")
                    st.write(f"**Similarity Score:** {similarity_score:.4f}")

                    similar_paper = df[df['arxiv_id'] == similar_paper_id].iloc[0]
                    st.write("### 📄 Most Similar Paper:")
                    st.write(f"**Title**: {similar_paper['title']}")
                    st.write(f"**Authors**: {similar_paper['authors']}")
                    st.markdown(f"[View Paper]({similar_paper['abs_url']})")

                    community_papers = get_community_papers(df, user_community, G)
                    st.write(f"### 📚 Papers in **Community {user_community}**:")
                    st.dataframe(community_papers[['arxiv_id', 'title', 'authors']].head(10))

                    user_node_id = f"user_{uuid.uuid4()}"  # Unique ID
                    G.add_node(user_node_id, title=user_title, authors="You", abstract=user_abstract,
                               community=user_community)

                    G.add_edge(user_node_id, similar_paper_id, weight=similarity_score)

                    partition[user_node_id] = user_community
                    nx.set_node_attributes(G, partition, 'community')

                    st.session_state.G = G
                    st.session_state.partition = partition

                    st.session_state.user_nodes.append(user_node_id)

                    st.success("🔗 Your article has been added to the graph.")

        st.header("🔍 View Graphs")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Show Updated Graph with Your Article", key="show_updated_graph"):
                if not st.session_state.user_nodes:
                    st.warning("⚠️ No user-submitted articles to show.")
                else:
                    with st.spinner('📈 Generating updated graph...'):
                        fig = plot_graph_plotly(G, partition, user_nodes=st.session_state.user_nodes,
                                                graph_type='updated')
                        st.plotly_chart(fig, use_container_width=True)

        with col2:
            if st.button("🌟 Show Community Graph", key="show_community_graph"):
                if not st.session_state.user_nodes:
                    st.warning("⚠️ No user-submitted articles to show.")
                else:
                    latest_user_node = st.session_state.user_nodes[-1]
                    user_community = partition.get(latest_user_node, -1)
                    if user_community == -1:
                        st.warning("⚠️ The latest user article could not be assigned to a community.")
                    else:
                        with st.spinner('📈 Generating community graph...'):
                            community_graph = G.subgraph(
                                [node for node, data in G.nodes(data=True) if data.get('community') == user_community]
                            )
                            if len(community_graph.nodes()) == 0:
                                st.warning("⚠️ No papers found in this community.")
                            else:
                                # Collect user nodes in the community
                                user_nodes_in_community = [node for node in st.session_state.user_nodes if
                                                           partition.get(node) == user_community]
                                fig_community = plot_graph_plotly(
                                    community_graph,
                                    {node: partition[node] for node in community_graph.nodes()},
                                    user_nodes=user_nodes_in_community,
                                    graph_type='community'
                                )
                                st.plotly_chart(fig_community, use_container_width=True)


if __name__ == "__main__":
    main()
