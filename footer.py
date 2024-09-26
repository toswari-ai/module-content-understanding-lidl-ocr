
# lifted from https://discuss.streamlit.io/t/st-footer/6447/25

def footer(st):
    with open('footer.html', 'r') as file:
        footer = file.read()
        st.write(footer, unsafe_allow_html=True)

