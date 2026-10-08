"""Pull every game row out of the season tabs of NHL.xlsx and match it to NHL game IDs.

Usage: python3 nhl_model/prepare_sheet.py path/to/NHL.xlsx
Writes nhl_model/data/games.csv (sheet ratings, lines, bets + final scores).
"""
import csv, datetime as dt, json, os, sys
import openpyxl
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")


def read_sheets(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    return {ws.title: [["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)]
            for ws in wb.worksheets if ws.title.endswith("Season")}


def extract(sheets):
    from openpyxl.utils import column_index_from_string as C
    def num(v):
        try: return float(v)
        except: return None
    layouts={
     '2022-2023_Season':dict(away='B',ab2b='C',agoal='E',hb2b='G',home='I',hgoal='J',ol_a='L',ol_h='M',pa='N',ph='O',aw='P',hw='Q',fa='AE',fh='AF',tvw='AL',bl='AO',risk='AU',wl='AW',tb='AX',betl='AY'),
     '2023-2024_Season':dict(away='B',ab2b='C',agoal='E',hb2b='H',home='J',hgoal='K',ol_a='N',ol_h='O',pa='P',ph='Q',uba='R',ubh='S',aw='T',hw='U',sa='AL',sh='AM',fa='AO',fh='AP',tvw='AV',bl='AZ',risk='BF',wl='BH',tb='BI',betl='BJ'),
    }
    layouts['2024-2025_Season']=layouts['2023-2024_Season']
    rows=[]
    for s,L in layouts.items():
        date=None
        for i,r in enumerate(sheets[s.replace('_',' ')]):
            if i==0: continue
            g=lambda k: r[C(L[k])-1].strip() if k in L and C(L[k])-1<len(r) else ''
            if r[0]:
                try: date=dt.datetime.strptime(r[0][:10],'%Y-%m-%d').date()
                except: pass
            if not g('away') or not g('home') or len(g('away'))>4: continue
            d=dict(sheet=s[:9],row=i+1,date=date,away=g('away'),home=g('home'),agoal=g('agoal'),hgoal=g('hgoal'),
                   ab2b=g('ab2b').lower()=='y',hb2b=g('hb2b').lower()=='y')
            for k in ['ol_a','ol_h','pa','ph','uba','ubh','aw','hw','sa','sh','fa','fh','bl','risk','betl']: d[k]=num(g(k))
            d['tvw']=g('tvw'); d['wl']=g('wl').lower(); d['tb']=g('tb')
            rows.append(d)
    df=pd.DataFrame(rows)
    print(df.groupby('sheet').agg(n=('away','size'),aw=('aw','count'),pa=('pa','count'),uba=('uba','count'),fa=('fa','count'),risk=('risk','count'),date=('date','count'),dmin=('date','min'),dmax=('date','max')))
    print(df.wl.value_counts().head(10))
    return df


def match(s):
    n=pd.DataFrame(json.load(open(os.path.join(DATA,'nhl_games.json')))); n['date']=pd.to_datetime(n.date)
    alias={'LA':'LAK','TB':'TBL','NJ':'NJD','SJ':'SJS','VEG':'VGK','WAS':'WSH','MON':'MTL','CLS':'CBJ','NAS':'NSH','CAL':'CGY','UTAH':'UTA','ARZ':'ARI','WPJ':'WPG','EMD':'EDM','WGP':'WPG'}
    for c in ['away','home']: s[c]=s[c].str.upper().replace(alias)
    print(set(s.away)-set(n.away))
    key=n.set_index(['away','home','date']).id.to_dict()
    ids=[]
    for r in s.itertuples():
        m=None
        for off in [0,-1,1,-2,2,-3,3,-4,4,-5,5,-6,6]:
            m=key.get((r.away,r.home,r.date+pd.Timedelta(days=off)))
            if m: break
        ids.append(m)
    s['gameId']=ids
    print(s.groupby('sheet').gameId.count(), s.gameId.isna().sum())
    print(s[s.gameId.isna()].groupby('sheet').date.agg(['min','max','count']))
    print('dupes',s.gameId.dropna().duplicated().sum())
    n = pd.DataFrame(json.load(open(os.path.join(DATA, "nhl_games.json"))))
    m = s.merge(n[["id", "ag", "hg", "last"]], left_on="gameId", right_on="id", how="left")
    m["homewin"] = (m.hg > m.ag).astype(float)
    m.loc[m.hg.isna(), "homewin"] = None
    return m


if __name__ == "__main__":
    s = extract(read_sheets(sys.argv[1]))
    s["date"] = pd.to_datetime(s.date)
    match(s).to_csv(os.path.join(DATA, "games.csv"), index=False)
