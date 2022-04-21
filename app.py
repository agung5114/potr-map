import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import json
import pickle 
from st_aggrid import AgGrid
import pandas as pd
from PIL import Image

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

st.set_page_config(layout="wide")
import streamlit.components.v1 as components

from googletrans import Translator, constants
from requests_html import HTMLSession
session = HTMLSession()
# from pygooglenews import GoogleNews
from pprint import pprint
import time
import pandas as pd
translator = Translator(service_urls=["translate.google.com"])
def translate(rawtext,src,dest):
    ts = translator.translate(rawtext,src=src,dest=dest)
    time.sleep(0.5)
    return ts.text

def get_titletext(keyword,src,dest,country_id):
    topik = translate(keyword,dest,src)
    url = f'https://news.google.com/rss/search?q={topik}&hl={src}&gl={country_id}&ceid={country_id}%3Aid'
    r = session.get(url)
    newstitles = []
    links = []
    for title in r.html.find('title')[:10]:
      newstitles.append(title.text)
      time.sleep(0.1)
    for link in r.html.find('description')[:10]:
      links.append(link.text)
      time.sleep(0.1)
    texts = [i.split(" - ", 1)[0] for i in newstitles]
    sources = [i.split(" - ", 1)[1] for i in newstitles]
    links = [i.split(" ", 2)[1] for i in links]
    links = [w[6:-1] for w in links]
    df = pd.DataFrame(list(zip(texts,sources,links)),columns =['News_Title','Source','Link'])
    df['News_Title'] = df['News_Title'].apply(translate,args=(src,dest))
    df['Country_ID'] = country_id
    df['Keyword'] = topik
    return df.iloc[1:, :]
# def get_titletext(keyword,src,dest,country_id):
#     gn = GoogleNews(lang = src, country = country_id)
#     search = gn.search(str(translate(keyword,dest,src)))
#     stories = []
#     links = []
#     newsitem = search['entries']
#     for item in newsitem[:10]:
#         stories.append(item.title)
#         links.append(item.link)
#     Original_Title, Source = map(list, zip(*(s.split("-") for s in stories)))
#     df = pd.DataFrame(list(zip(Original_Title, Source,links)),columns =['Original_Title', 'Source','Link'])
#     df['Keyword'] = keyword
#     df['Country_ID'] = country_id
#     df['Title_in_English'] = df['Original_Title'].apply(translate,args=(src,dest))
#     return df[['Title_in_English','Keyword','Source','Link']]
def nlp_unspv(df,column,cluster):
    base = df[column].astype('str')
    # base = df[column]
    v = TfidfVectorizer()
    x = v.fit_transform(base)
    clf = TruncatedSVD(3)
    Xpca = clf.fit_transform(x)
    kmeans = KMeans(n_clusters=cluster).fit(Xpca)
    df['Cluster'] = kmeans.predict(Xpca)
    return df

##TOP PAGE
st.title("Palm Oils Trafficking Risk Mapping Dashboard")
st.markdown('<style>h1{color:dark-grey;font-size:62px}</style>',unsafe_allow_html=True)
st.sidebar.image(Image.open('logo.png'))
st.sidebar.image(Image.open('text.png'))
menu = ["TAHUB DATA","CLUSTERING","NEWS SCRAPER","FINANCIAL ANALYSIS", "RISK MAPS"]
choice = st.sidebar.selectbox("Select Menu", menu)
if choice == "TAHUB DATA":
    components.html('''
        <div class='tableauPlaceholder' id='viz1650451818109' style='position: relative'><noscript><a href='#'><img alt='Number of Incident Reporting ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Tr&#47;TraffickingIncidentReporting&#47;NumberofIncidentReporting&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TraffickingIncidentReporting&#47;NumberofIncidentReporting' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Tr&#47;TraffickingIncidentReporting&#47;NumberofIncidentReporting&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div> <script type='text/javascript'> var divElement = document.getElementById('viz1650451818109');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1366px';vizElement.style.height='795px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1366px';vizElement.style.height='795px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
        ''',height=1150,
            width=1680)
       
elif choice == "CLUSTERING":
    st.subheader('PCA Table')
    df = pd.read_feather('tahub_data.feather')
    df = df[["Country","Trafficking Type","Industry Sector","Coercion Method","Victim Gender","Victim Age","Incident Reporting Date"]]
    AgGrid(df)
    st.subheader('Clustering')
    cl_exp = st.expander(label='Clustering Parameters')
    with cl_exp:
        Cluster = st.number_input('Number of Cluster',min_value=1,max_value=100,step=1)
        Column = st.selectbox("Column Title",df.columns)
        Country = st.multiselect("Select Country",df.Country.unique())
        if st.button("Run Clustering"):
            st.write("Clustering Output")
            df_new = df[df['Country'].isin(Country)]
            df_new = nlp_unspv(df_new,Column, Cluster)
            df_new['count'] = 1
            df_plot = df_new.groupby(by=['Country','Cluster'],as_index=False).agg({'count':'sum'})
            fig = px.bar(df_plot,x='Cluster',y='count',color='Country',barmode='stack')
            st.plotly_chart(fig)
            AgGrid(df_new)
elif choice =="NEWS SCRAPER":
    df = pd.read_csv('scrap_keywords.csv')
    st.subheader('News Related with Palm Oil Workers')
    AgGrid(df)
    kibana = st.expander(label='Kibana Dashboard')
    with kibana:
        components.html('''
        <style>
        #wrap { width: 1020px; height: 900px; padding: 0; border: 0px solid grey; overflow: hidden; }
        #frame { width: 1680px; height: 900px;padding: 0; margin-top: -56px; border: 0px solid grey; overflow: hidden;}
        </style>
        <iframe id="frame" src="https://potr-map.kb.us-central1.gcp.cloud.es.io:9243/app/dashboards#/view/5eb245a0-c0c3-11ec-beee-630d91440ab3?embed=true&_g=(filters:!(),refreshInterval:(pause:!t,value:0),time:(from:now-1y%2Fd,to:now))&_a=(description:'NewsScrap%20about%20Palm%20Oil%20Worker!'s%20Explotiatation',filters:!(),fullScreenMode:!f,options:(hidePanelTitles:!f,syncColors:!f,useMargins:!t),panels:!((embeddableConfig:(attributes:(references:!((id:'9d215a30-c053-11ec-beee-630d91440ab3',name:indexpattern-datasource-current-indexpattern,type:index-pattern),(id:'9d215a30-c053-11ec-beee-630d91440ab3',name:indexpattern-datasource-layer-ebcdb1fe-84e4-4413-a32d-e47305bbf400,type:index-pattern)),state:(datasourceStates:(indexpattern:(layers:(ebcdb1fe-84e4-4413-a32d-e47305bbf400:(columnOrder:!(fda240ea-415e-4875-8524-4956501a0edd),columns:(fda240ea-415e-4875-8524-4956501a0edd:(dataType:number,isBucketed:!f,label:'Unique%20count%20of%20Link',operationType:unique_count,scale:ratio,sourceField:Link)),incompleteColumns:())))),filters:!(),query:(language:kuery,query:'palm%20oil'),visualization:(accessor:fda240ea-415e-4875-8524-4956501a0edd,layerId:ebcdb1fe-84e4-4413-a32d-e47305bbf400,layerType:data)),title:'',type:lens,visualizationType:lnsMetric),enhancements:(),hidePanelTitles:!f),gridData:(h:12,i:db0d4e59-fd43-4ce9-aab4-a62a2197cd1d,w:11,x:0,y:0),panelIndex:db0d4e59-fd43-4ce9-aab4-a62a2197cd1d,title:'Palm%20Oil%20Headline',type:lens,version:'8.1.2'),(embeddableConfig:(attributes:(references:!((id:'9d215a30-c053-11ec-beee-630d91440ab3',name:indexpattern-datasource-current-indexpattern,type:index-pattern),(id:'9d215a30-c053-11ec-beee-630d91440ab3',name:indexpattern-datasource-layer-0dddb5e7-dd3c-457b-b0c9-4509a64671c6,type:index-pattern),(id:'9d215a30-c053-11ec-beee-630d91440ab3',name:e5cd7fac-3db2-4bf2-a6f7-a9cf69965a41,type:index-pattern)),state:(datasourceStates:(indexpattern:(layers:('0dddb5e7-dd3c-457b-b0c9-4509a64671c6':(columnOrder:!('8611b55b-aaee-456e-9d36-294461741af3',e0fe4b9a-3909-45d4-9011-cbd0f7f3006a,'30f11861-d7e8-4c70-8637-3fc591e838ca','67cf014f-19ec-40f9-801d-cabb480445da'),columns:('30f11861-d7e8-4c70-8637-3fc591e838ca':(dataType:string,isBucketed:!t,label:'Top%20values%20of%20Link',operationType:terms,params:(missingBucket:!f,orderBy:(columnId:'67cf014f-19ec-40f9-801d-cabb480445da',type:column),orderDirection:desc,otherBucket:!t,parentFormat:(id:terms),size:3),scale:ordinal,sourceField:Link),'67cf014f-19ec-40f9-801d-cabb480445da':(dataType:number,isBucketed:!f,label:'Count%20of%20records',operationType:count,scale:ratio,sourceField:___records___),'8611b55b-aaee-456e-9d36-294461741af3':(dataType:string,isBucketed:!t,label:'Top%20values%20of%20Country_ID',operationType:terms,params:(missingBucket:!f,orderBy:(columnId:'67cf014f-19ec-40f9-801d-cabb480445da',type:column),orderDirection:desc,otherBucket:!t,parentFormat:(id:terms),size:5),scale:ordinal,sourceField:Country_ID),e0fe4b9a-3909-45d4-9011-cbd0f7f3006a:(dataType:string,isBucketed:!t,label:'Top%20values%20of%20Keyword',operationType:terms,params:(missingBucket:!f,orderBy:(columnId:'67cf014f-19ec-40f9-801d-cabb480445da',type:column),orderDirection:desc,otherBucket:!t,parentFormat:(id:terms),size:3),scale:ordinal,sourceField:Keyword)),incompleteColumns:())))),filters:!(('$state':(store:appState),meta:(alias:!n,disabled:!f,index:e5cd7fac-3db2-4bf2-a6f7-a9cf69965a41,key:Keyword,negate:!f,params:(query:'worker%20exploitation'),type:phrase),query:(match_phrase:(Keyword:'worker%20exploitation')))),query:(language:kuery,query:''),visualization:(layers:!((categoryDisplay:default,groups:!('8611b55b-aaee-456e-9d36-294461741af3',e0fe4b9a-3909-45d4-9011-cbd0f7f3006a,'30f11861-d7e8-4c70-8637-3fc591e838ca'),layerId:'0dddb5e7-dd3c-457b-b0c9-4509a64671c6',layerType:data,legendDisplay:default,metric:'67cf014f-19ec-40f9-801d-cabb480445da',nestedLegend:!f,numberDisplay:percent)),shape:pie)),title:'',type:lens,visualizationType:lnsPie),enhancements:(),hidePanelTitles:!f),gridData:(h:12,i:e9cb8515-6b75-47e4-9381-c1c224cd6849,w:12,x:11,y:0),panelIndex:e9cb8515-6b75-47e4-9381-c1c224cd6849,title:'By%20Country',type:lens,version:'8.1.2'),(embeddableConfig:(attributes:(references:!((id:'9d215a30-c053-11ec-beee-630d91440ab3',name:indexpattern-datasource-current-indexpattern,type:index-pattern),(id:'9d215a30-c053-11ec-beee-630d91440ab3',name:indexpattern-datasource-layer-940279c2-27c2-410e-8c6b-60fce6cd4269,type:index-pattern)),state:(datasourceStates:(indexpattern:(layers:('940279c2-27c2-410e-8c6b-60fce6cd4269':(columnOrder:!(d14aa505-b737-4a9f-9021-f36f0b4cb756,'9e467722-bc78-429b-a5b7-24348a64ed40'),columns:('9e467722-bc78-429b-a5b7-24348a64ed40':(dataType:number,isBucketed:!f,label:'Unique%20count%20of%20Keyword',operationType:unique_count,scale:ratio,sourceField:Keyword),d14aa505-b737-4a9f-9021-f36f0b4cb756:(dataType:string,isBucketed:!t,label:'Top%20values%20of%20Link',operationType:terms,params:(missingBucket:!f,orderBy:(columnId:'9e467722-bc78-429b-a5b7-24348a64ed40',type:column),orderDirection:desc,otherBucket:!t,parentFormat:(id:terms),size:15),scale:ordinal,sourceField:Link)),incompleteColumns:())))),filters:!(),query:(language:kuery,query:''),visualization:(columns:!((columnId:d14aa505-b737-4a9f-9021-f36f0b4cb756,width:796.5),(columnId:'9e467722-bc78-429b-a5b7-24348a64ed40',isTransposed:!f)),layerId:'940279c2-27c2-410e-8c6b-60fce6cd4269',layerType:data)),title:'',type:lens,visualizationType:lnsDatatable),enhancements:(),hidePanelTitles:!f),gridData:(h:22,i:d68015cc-19b1-4419-9ab6-dea16d27cd2d,w:25,x:23,y:0),panelIndex:d68015cc-19b1-4419-9ab6-dea16d27cd2d,title:'Source%20URLS',type:lens,version:'8.1.2'),(embeddableConfig:(attributes:(references:!((id:'9d215a30-c053-11ec-beee-630d91440ab3',name:indexpattern-datasource-current-indexpattern,type:index-pattern),(id:'9d215a30-c053-11ec-beee-630d91440ab3',name:indexpattern-datasource-layer-29709d1e-17bd-43f1-b778-0aab9a64fa01,type:index-pattern)),state:(datasourceStates:(indexpattern:(layers:('29709d1e-17bd-43f1-b778-0aab9a64fa01':(columnOrder:!(b8f8753c-57cb-4112-85e2-32cf04af0a62,'307ceb17-3fbe-44a1-8167-48f12243eb99'),columns:('307ceb17-3fbe-44a1-8167-48f12243eb99':(dataType:number,isBucketed:!f,label:'Count%20of%20records',operationType:count,scale:ratio,sourceField:___records___),b8f8753c-57cb-4112-85e2-32cf04af0a62:(dataType:string,isBucketed:!t,label:'Top%20values%20of%20Keyword',operationType:terms,params:(missingBucket:!f,orderBy:(columnId:'307ceb17-3fbe-44a1-8167-48f12243eb99',type:column),orderDirection:desc,otherBucket:!t,parentFormat:(id:terms),size:5),scale:ordinal,sourceField:Keyword)),incompleteColumns:())))),filters:!(),query:(language:kuery,query:''),visualization:(layers:!((categoryDisplay:default,groups:!(b8f8753c-57cb-4112-85e2-32cf04af0a62),layerId:'29709d1e-17bd-43f1-b778-0aab9a64fa01',layerType:data,legendDisplay:default,metric:'307ceb17-3fbe-44a1-8167-48f12243eb99',nestedLegend:!f,numberDisplay:percent)),shape:treemap)),title:'',type:lens,visualizationType:lnsPie),enhancements:(),hidePanelTitles:!f),gridData:(h:10,i:'04174d6d-06ad-45f2-9913-1dd9c013b1e9',w:23,x:0,y:12),panelIndex:'04174d6d-06ad-45f2-9913-1dd9c013b1e9',title:'By%20Keyword',type:lens,version:'8.1.2')),query:(language:kuery,query:''),tags:!(),timeRestore:!f,title:NewsScrap,viewMode:edit)&show-query-input=true"
        ></iframe>
        ''',height=900,width=1680)
    st.subheader('Custom News Scraping')
    custom_search = st.expander(label='Search Parameters')
    with custom_search:
        keyword = st.text_input("Search Keyword")
        slang = st.text_input("Source Language ID")
        ctr_id = st.text_input("Country ID")
        if st.button("Run Scraping"):
            df_scrap = get_titletext(keyword.lower(),slang.lower(),'en',ctr_id.upper())
            AgGrid(df_scrap)
elif choice =="FINANCIAL ANALYSIS":
    df = pd.read_csv('cpodata.csv')
    st.subheader('Worker Exploitation by Ratio Indicators')
    AgGrid(df)

elif choice =="RISK MAPS":
    components.html('''
        <div class='tableauPlaceholder' id='viz1650513950841' style='position: relative'><noscript><a href='#'><img alt='Geographic-Specific Risk Profile ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pa&#47;PalmOilIndustry-RiskAnalysis&#47;Geographic-SpecificRiskProfile&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='PalmOilIndustry-RiskAnalysis&#47;Geographic-SpecificRiskProfile' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Pa&#47;PalmOilIndustry-RiskAnalysis&#47;Geographic-SpecificRiskProfile&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1650513950841');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1024px';vizElement.style.height='795px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1024px';vizElement.style.height='795px';} else { vizElement.style.width='100%';vizElement.style.height='1877px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
        ''',height=795,
            width=1280)
    maps_exp = st.expander('Palm oil mills maps')
    with maps_exp:
        components.html('''
        <style>
        #wrap { width: 1020px; height: 900px; padding: 0; border: 0px solid grey; overflow: hidden; }
        #frame { width: 1680px; height: 900px;padding: 0; margin-top: -56px; border: 0px solid grey; overflow: hidden;}
        </style>
        <iframe id="frame" scrolling="no" class="wrapped-iframe" gesture="media"  allow="encrypted-media" allowfullscreen = "True"
        name="Framename" sandbox="allow-same-origin allow-scripts allow-popups allow-forms" 
        src="https://www.arcgis.com/home/webmap/viewer.html?useExisting=1&layers=3b28b8bcc5144cb685eb397979ea602f"
        style="width: 100%;">
        </iframe>
        '''
        ,height=795,
        width=1150)
    

    # txt = st.text_input('Input Text Here')
    # st.write(f'ini inputan text:  {txt}')
    # nbr = st.number_input('Input Number Here')
    # st.write(f'ini inputan number:  {nbr}')
    # image = st.file_uploader(label='File Uploader')
    # col1, col2 = st.columns(2)

    # if image == None:
    #     st.write('Please Upload an Image')
    # else:
    #     original = Image.open(image)
    #     col1.header("Original")
    #     col1.image(original, use_column_width=True)

    #     grayscale = original.convert('LA')
    #     col2.header("Grayscale")
    #     col2.image(grayscale, use_column_width=True)

    # # Expander
    # st.title("Expander")
    # my_expander = st.expander(label='Expander1')
    # with my_expander:
    #     'Hello there!'
    #     c1, c2, c3, c4 = st.columns((2, 1, 1, 1))
    #     with c1:
    #         "kolom 1"
    #     with c2:
    #         "kolom 2"
    #     with c3:
    #         "kolom 3"
    #     with c4:
    #         "kolom 4"
