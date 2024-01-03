import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#primary elo function variables
L=350 #default 500
k=.0005 #default .0005
baseelo = 2000

#bracket weight variables
x0=0 #default 0
L1 = 5 #default 3.5
k1 = .0013 #default .0013
bracketvalues1 = [1500, 3000, 4000] #default [1500, 3000, 4000]
bracketvalues2 = [1000, 2000, 3000, 4000] #default [1000, 2000, 3000, 4000]

#decay variables
x02 = 15 #default 18 (lower number results in stronger decay)
xn = 2 #default 1

#ranks (lol)
start = 0
interval = 500
div = 2
ranks = ['D', 'C', 'C+', 'B', 'B+', 'A', 'A+', 'S', 'S+', 'PFC', 'PFC+','erm']

bracketadjust = True
mmradjujst = False
decayadjust = True
peaktracking = True
ranktracking = True

elo = pd.read_csv("elo.csv")
mmr = pd.read_csv("mmr.csv")
matches0 = pd.read_csv("matches0.csv")
matches = pd.read_csv("matches.csv")
matches2 = pd.read_csv("matches2.csv")
matches3 = pd.read_csv("matches3.csv")
matches4 = pd.read_csv("matches4.csv")
matches5 = pd.read_csv("matches5.csv")
matches6 = pd.read_csv("matches6.csv") 
matches7 = pd.read_csv("matches7.csv")
matches8 = pd.read_csv("matches8.csv")
matches9 = pd.read_csv("matches9.csv")
elodf = pd.DataFrame(data = elo)
mmrdf = pd.DataFrame(data = mmr)
matches = [matches9, matches8, matches7, matches6, matches5, matches4, matches3, matches2, matches, matches0]

def decayfunc(elo, x):
  return((elo * x02) / x)

def match(list):
  elo1 = elodf.loc[elodf["Name"] == list[0]].iloc[0]['Elo']
  elo2 = elodf.loc[elodf["Name"] == list[0]].iloc[0]['Elo']
  return(
    L/(1 + np.exp(-k * ((elo1 - elo2) - x0))) -  
    500 +
    500 * (list[2] / (list[2] + list[3]))
  )

def multiplier(diff):
  m = (1 - L1) / ((L1 / 2) - L1)
  m1 = (2 / L1) - (2 / (L1 * L1))
  C = L1 - m * L1
  C1 = 1 / L1
  if diff <= 0:
    return(
      C1 + (m1 * L1) / (1 + np.exp(-k1 * diff))
    )
  if diff > 0:
    return(
      C + (m * L1) / (1 + np.exp(-k1 * diff))
    )

def modifier(elo, mmr, score):
  if score == 0:
    return(0)
  else:
    return(
      score * multiplier(
        (abs(score) / score) * (mmr - elo)
      )
    )

def decay():
  for index, row in mmrdf.loc[mmrdf["MMR"] != 0].iterrows():
    elodf.iloc[index] = [elodf.iloc[index]["Name"], decayfunc(elodf.iloc[index]["Elo"], x02 + xn), elodf.iloc[index]["Peak"], elodf.iloc[index]['Rank'], elodf.iloc[index]['Peak Rank']]

if ranktracking == True:
  def getrank(elo):
    return(ranks[np.floor(elo/interval).astype(int)] #+ ' ' + str(
    #((elo - interval * np.floor(elo/interval) + interval / div - (elo - interval * np.floor(elo / interval)) % (interval / div)) * div / interval).astype(int)
  #)
          )
else:
  def getrank(elo):
    return('N/A')

if bracketadjust == True:
  def matchelo(P1, P2, S1, S2):
    if elodf.loc[elodf["Name"] == P1].iloc[0]['Elo'] == 0:
      elodf.loc[elodf["Name"] == P1] = [P1, baseelo, elodf.loc[elodf["Name"] == P1].iloc[0]['Peak'], elodf.loc[elodf["Name"] == P1].iloc[0]['Rank'], elodf.loc[elodf["Name"] == P1].iloc[0]['Peak Rank']]
    if elodf.loc[elodf["Name"] == P2].iloc[0]['Elo'] == 0:
      elodf.loc[elodf["Name"] == P2] = [P2, baseelo, elodf.loc[elodf["Name"] == P2].iloc[0]['Peak'], elodf.loc[elodf["Name"] == P2].iloc[0]['Rank'], elodf.loc[elodf["Name"] == P2].iloc[0]['Peak Rank']]
    elo1 = elodf.loc[elodf["Name"] == P1].iloc[0]['Elo']
    elo2 = elodf.loc[elodf["Name"] == P2].iloc[0]['Elo']
    newelo1 = elo1 + modifier(elo1, mmrdf.loc[mmrdf["Name"] == P1]["MMR"].values[0],  match([P1, P2, S1, S2]))
    newelo2 = elo2 + modifier(elo2, mmrdf.loc[mmrdf["Name"] == P2]["MMR"].values[0], -match([P1, P2, S1, S2]))
    elodf.loc[elodf["Name"] == P1] = [P1, newelo1, elodf.loc[elodf["Name"] == P1].iloc[0]['Peak'], elodf.loc[elodf["Name"] == P1].iloc[0]['Rank'], elodf.loc[elodf["Name"] == P1].iloc[0]['Peak Rank']]
    elodf.loc[elodf["Name"] == P2] = [P2, newelo2, elodf.loc[elodf["Name"] == P2].iloc[0]['Peak'], elodf.loc[elodf["Name"] == P2].iloc[0]['Rank'], elodf.loc[elodf["Name"] == P2].iloc[0]['Peak Rank']]
else:
  def matchelo(P1, P2, S1, S2):
    elodf.loc[elodf["Name"] == P1] = [P1, elodf.loc[elodf["Name"] == P1].iloc[0]['Elo'] + match([P1, P2, S1, S2])]
    elodf.loc[elodf["Name"] == P2] = [P2, elodf.loc[elodf["Name"] == P2].iloc[0]['Elo'] - match([P1, P2, S1, S2])]

brackets1 = ['Fails','Close','Playoffs']
brackets2 = ['Fails','Close','Challengers','Elites']
graphpoints = []
ssso = 1

for list in matches:
  print(ssso)
  if decayadjust == True:
    print('you just got decayed ! lol')
    decay()

  if decayadjust == True or bracketadjust == True:
    if ssso <= 6:
      brackets, bracketvalues = brackets1, bracketvalues1
    else:
      brackets, bracketvalues = brackets2, bracketvalues2
    n=0
    for bracket in brackets:
      for player in list[bracket]:
        if player == '1':
          continue
        mmrdf.loc[mmrdf["Name"] == player] = [player, bracketvalues[n]]
      n += 1
  for index, row in list.iterrows():
    matchelo(row['P1'], row['P2'], row['S1'], row['S2'])
  graphpoints.append(elodf['Elo'].values.tolist())
  if peaktracking == True:
    for index, row in elodf.iterrows():
      if row['Elo'] == 2000:
        continue
      elodf.iloc[index] = [row['Name'], row['Elo'], max(row['Elo'], row['Peak']), getrank(row['Elo']), getrank(max(row['Elo'], row['Peak']))]
  ssso += 1
  
print(elodf[mmrdf['MMR'] != 0].sort_values(by = ['Elo'], ascending=False).reset_index().head(50))

import datetime as dt
dates = ["2020-05-02","2020-08-17","2020-11-07", "2021-01-17", "2021-05-09", "2021-08-15", "2022-01-30", "2022-07-31", "2023-01-29", "2023-06-25"]
dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
graphpoints = np.array(graphpoints).T
graphpoints = np.flipud((graphpoints[np.argsort(graphpoints[:, 9])])).tolist()
n=0
for list in graphpoints:
  if elodf.sort_values(by = 'Elo', ascending=False).iloc[n]["Elo"] > 2500:
    plt.plot(dates, list, label = elodf.sort_values(by = 'Elo', ascending=False).iloc[n]["Name"])
    plt.text(dates[list.index(max(list))], list[list.index(max(list))], elodf.sort_values(by = 'Elo', ascending=False).iloc[n]["Name"])
  n+=1


plt.axhspan(4500,5500, facecolor = 'magenta', alpha = .3)
plt.axhspan(3500,4500, facecolor = 'aqua', alpha = .3)
plt.axhspan(2500,3500, facecolor = 'chartreuse', alpha = .3)
plt.axhspan(1500,2500, facecolor = 'yellow', alpha = .3)
plt.axhspan(500,1500, facecolor = 'orange', alpha = .3)
plt.axhspan(0,500, facecolor = 'darkgoldenrod', alpha = .3)
plt.ylim(30,5500)
elodf.to_csv('elofinal', index = False)
plt.legend(bbox_to_anchor=(1, .5), loc="center left")
plt.show()