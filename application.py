import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Stock Analysis",initial_sidebar_state="expanded"
)

st.title('Tensors & RL with Stocks')

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if 'predictions' not in st.session_state:
    predictions = pd.read_csv('predictions.csv',index_col=0)
    predictions['Model Run Date'] = pd.to_datetime(predictions['Model Run Date'])
    uniqStocks = set(predictions['Stock'])
    st.session_state.predictions = predictions
    st.session_state.uniqStocks = list(uniqStocks)

getStock = st.selectbox(
        "Select a stock symbol.",
        st.session_state.uniqStocks,
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

# Do highlights later
subsetFrame = st.session_state.predictions[st.session_state.predictions.Stock == getStock]

st.dataframe(subsetFrame)

chart = alt.Chart(subsetFrame)\
    .mark_line().encode(
    alt.Y('Actual Close Price On Model Run Date')\
        .scale(domain=(subsetFrame['Actual Close Price On Model Run Date'].min()-15, 
                       subsetFrame['Actual Close Price On Model Run Date'].max()+15)),
    x='Model Run Date'
).interactive()

subsetFrameAnnBuy = subsetFrame[subsetFrame.Prediction=='Buy']
subsetFrameAnnSell = subsetFrame[subsetFrame.Prediction=='Sell']

annotation_layer_buy = alt.Chart(subsetFrameAnnBuy)\
    .mark_text(size=20, text="⬇", dx=-8, dy=-10, align="left",color='green')\
    .encode(
        alt.Y('Actual Close Price On Model Run Date'),
        x='Model Run Date'
    ).interactive()

annotation_layer_sell = alt.Chart(subsetFrameAnnSell)\
    .mark_text(size=20, text="⬇", dx=-8, dy=-10, align="left",color='red')\
    .encode(
        alt.Y('Actual Close Price On Model Run Date'),
        x='Model Run Date'
    ).interactive()


st.altair_chart(chart + annotation_layer_buy + annotation_layer_sell, theme="streamlit", use_container_width=True)



