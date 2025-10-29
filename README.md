#  RouteLens — Flight Delay Exploration

Interactive flight delay exploration using US BTS On-Time Performance data.  
Built with **Streamlit + Pandas + Plotly + PyDeck**.

This app lets you:
Select date / origin / destination  
Visualize flight delay patterns on a map  
Explore heatmaps by departure time  
View delay category breakdowns  

---

##  Project Structure
project/
│
├── pds.py # Main Streamlit app
├── requirements.txt
│
├── data/
│ └── 2024_07_sample.csv.gz # Sample BTS flight data (~100k rows)
│
└── flight_geo/
└── L_AIRPORT_ID_with_Coordinates.csv # Airport geolocation reference

##  Run Locally

### 1) Clone the repo
```bash
git clone https://github.com/flight-delay-xai.git
cd flight-delay-xai

pip install -r requirements.txt
streamlit run pds.py

