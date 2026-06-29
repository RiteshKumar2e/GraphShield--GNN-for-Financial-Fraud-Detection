# GraphShield — Graph Visualization Explained

## Graph kya dikhata hai?

Jab aap koi Node ID daalte ho aur **Score** dabate ho, dashboard ek **2-hop transaction subgraph** dikhata hai.

Iska matlab:
- **Center mein** woh transaction hoti hai jo aapne search ki (Target Node)
- **Uske aas-paas** woh transactions hoti hain jinhone usse Bitcoin bheja ya jinhe usne bheja (1-hop neighbours)
- **Usse aage** unke bhi neighbours dikhte hain (2-hop neighbours)

Yeh graph model ko yeh samajhne mein help karta hai ki koi transaction **kitni suspicious connections** rakhti hai — sirf khud ki features se nahi, balki apne **poore network context** se.

---

## Nodes (Circles) ka matlab

| Circle ka rang | Matlab | ID ke upar likha |
|---|---|---|
| 🔵 Blue (bada) | **Target Node** — woh transaction jo aapne search ki | `TARGET` |
| 🔴 Red | **Illicit Transaction** — confirmed fraud node | `ILLICIT` |
| 🟢 Green | **Licit Transaction** — confirmed clean node | `LICIT` |

### Circle ke andar kya likha hota hai?

Har circle ke andar **2 lines** hoti hain:

```
┌─────────────┐
│   TARGET    │  ← Upar: Node ka type (TARGET / ILLICIT / LICIT)
│    27695    │  ← Neeche: Node ka ID number
└─────────────┘
```

### Circle ke neeche % kya hai?

Har **red ya blue** node ke neeche ek percentage likhi hoti hai — yeh us node ki **fraud probability** hai.

```
     🔴
   ILLICIT
    28039
     91%      ← Model ne 91% chance diya ki yeh fraud hai
```

---

## Outer Ring (Arc) ka matlab

Har illicit ya target node ke bahar ek **arc ring** hoti hai:

```
    ╔══════════╗   ← Poori ring bhari = 100% fraud
    ║  ILLICIT ║
    ║  27695   ║
    ╚══════════╝

    ╔══╗           ← Aadhi ring = 50% fraud
    ║  ║
    ╚══╝

    ╔╗             ← Chhoti ring = 10% fraud
    ╚╝
```

- **Jitni zyada ring bhari** → utni zyada fraud probability
- **Poori ring** → model almost sure hai ki yeh fraud hai
- **Chhoti ring** → model ko doubt hai, borderline case

---

## Edges (Arrows) ka matlab

Arrow ka matlab hai **Bitcoin ka flow** — paise kahan se kahan gaye.

```
[Transaction A] ──▶ [Transaction B]
```
Matlab: Transaction A ne Bitcoin bheja Transaction B ko.

### Arrow colors

| Arrow ka rang | Matlab |
|---|---|
| 🔴 Red arrow | Bitcoin flow kisi **fraud node ke saath** connected hai |
| 🔵 Blue arrow | Bitcoin flow **target node** se directly connected hai |
| ⚫ Grey arrow | Normal flow — dono nodes clean hain |

### Arrow ki thickness

Arrow jitni **moti** hoti hai, utna **zyada weight** us edge ko model ne diya — matlab us connection ne prediction mein zyada role play kiya.

---

## Example: Node 27695 — Fraud Ring Hub

```
         28026 (LICIT)
           ↑
           |
28039 ──▶ 27695 (TARGET · 98.9% fraud) ──▶ 28579
(LICIT)    ↑                                (LICIT)
           |
         27832 (LICIT)
```

**Yahan kya ho raha hai:**
- Node 27695 ek **hub** hai — bahut saari transactions se connected hai
- Khud toh ILLICIT mark hai, lekin uske aas-paas ke nodes mostly LICIT hain
- Model ne iska fraud score 98.9% diya kyunki uski **khud ki features** (neighbourhood stats) fraud patterns match karti hain
- Yeh ek **money laundering hub** ka typical pattern hai — fraud node clean nodes ke through paise move karta hai

---

## Example: Normal Licit Node

```
    28026 (LICIT) ──▶ 28039 (LICIT) ──▶ 28579 (LICIT)
```

**Yahan kya ho raha hai:**
- Saare nodes green hain — normal transactions
- Grey arrows — clean money flows
- Koi fraud connection nahi
- Fraud probability bahut kam hogi (< 10%)

---

## Features ka matlab (Sidebar mein)

Jab aap kisi node ko score karte ho, **Top Predictive Features** mein bars aate hain:

| Feature prefix | Matlab |
|---|---|
| `local_0` to `local_93` | **Transaction-level stats** — amount, inputs count, outputs count, fees |
| `nbr_0` to `nbr_71` | **Neighbourhood aggregates** — 1-hop neighbours ke stats ka average |

**Agar `nbr_` features top pe hain:**
→ Model ne fraud **network structure** se pakda — fraud ring mein hai

**Agar `local_` features top pe hain:**
→ Model ne fraud **transaction behaviour** se pakda — amount ya pattern suspicious hai

---

## Inference: Model kaise decide karta hai?

```
Node ke apne features (166 features)
          +
Neighbours ke features (message passing)
          +
Neighbours ke neighbours ke features (2-hop)
          ↓
    GNN Model (GAT / GraphSAGE / GCN)
          ↓
    Fraud Probability (0% to 100%)
          ↓
    ILLICIT (≥ 50%) ya LICIT (< 50%)
```

### Confidence levels

| Confidence | Fraud Probability | Matlab |
|---|---|---|
| HIGH ILLICIT | ≥ 80% | Almost confirmed fraud |
| MEDIUM ILLICIT | 50–80% | Likely fraud, review karo |
| REVIEW | 30–50% | Suspicious, manual check chahiye |
| LICIT | < 30% | Likely clean |

---

## Kuch interesting nodes try karo

| Node ID | Type | Kyu interesting hai |
|---|---|---|
| `27695` | ILLICIT | Fraud hub — 30 connections, 98.9% fraud |
| `31473` | ILLICIT | Another hub — 28 connections |
| `18903` | ILLICIT | 26 connections |
| `7929`  | ILLICIT | Isolated fraud — koi neighbour nahi, sirf features se pakda |
| `100`   | LICIT | Normal clean transaction |

---

## Summary

```
Aap jo graph dekh rahe ho uska simple matlab:

  🔵 = Woh transaction jo aapne search ki
  🔴 = Confirmed fraud transactions jo connected hain
  🟢 = Clean transactions jo connected hain
  ──▶ = Bitcoin ka flow (paise gaye)
  Arc ring = Kitna confident hai model (jitni bhari ring, utna zyada fraud)
  % label = Exact fraud probability
  
  Agar target node ke paas zyada 🔴 nodes hain
  → Model fraud probability zyada dega
  → Yeh fraud ring ka sign hai
```
