# Symbol Configuration Guide
## Elliott Wave Trading Engine V2

### 📋 **So änderst du die Trading-Symbole:**

#### **1. Symbol-Dateien bearbeiten:**
- **`symbols.txt`** - Vollständige Symbol-Liste (17 Symbole)
- **`symbols_simple.txt`** - Einfache Liste (3 Symbole) 
- **`symbols_demo.txt`** - Demo Mode (1 Symbol)

#### **2. Eigene Symbol-Datei erstellen:**
```
# Meine Symbole
EURUSD
GBPUSD
XAUUSD
US30
```

#### **3. System starten:**

**Option A: Vordefinierte Listen**
```batch
start_full_trading.bat     # Alle 17 Symbole
start_simple_trading.bat   # 3 Symbole (EURUSD, XAUUSD, US30)
start_demo_trading.bat     # 1 Symbol (nur EURUSD)
```

**Option B: Eigene Datei**
```batch
python elliott_wave_trader_v2.py meine_symbole.txt
```

#### **4. Symbol-Datei Format:**
```
# Kommentare beginnen mit #
# Ein Symbol pro Zeile
# Leer Zeilen werden ignoriert

EURUSD     # Major Forex
XAUUSD     # Gold
US30       # Dow Jones

# Deaktivierte Symbole
# GBPUSD    # Auskommentiert
```

### 🎯 **Empfohlene Symbol-Listen:**

#### **Anfänger (sicher):**
```
EURUSD    # Meist liquide Forex
```

#### **Fortgeschritten:**
```
EURUSD    # Major Forex
XAUUSD    # Gold
US30      # US Index
```

#### **Profi:**
```
EURUSD    # Major Forex Pairs
GBPUSD
AUDUSD
XAUUSD    # Precious Metals
XAGUSD
US30      # US Indices  
NAS100
US500.f
```

### ⚠️ **Wichtige Hinweise:**

1. **Symbol-Namen** müssen **exakt** mit MT5 übereinstimmen
2. **Groß-/Kleinschreibung** wird automatisch korrigiert
3. **Nicht verfügbare Symbole** werden automatisch übersprungen
4. **Teste erst** mit wenigen Symbolen (Demo Mode)
5. **Mehr Symbole = mehr Risiko** (Portfolio-Diversifikation beachten)

### 🔄 **Symbole zur Laufzeit ändern:**
1. Symbol-Datei bearbeiten (z.B. `symbols.txt`)
2. System **neu starten**
3. Neue Symbole werden automatisch geladen

### 📊 **Symbol-Kategorien:**

| Kategorie | Symbole | Charakteristik |
|-----------|---------|---------------|
| **Major Forex** | EURUSD, GBPUSD, AUDUSD | Hohe Liquidität, niedrige Spreads |
| **Minor Forex** | AUDNOK, EURNOK | Mittlere Liquidität, höhere Volatilität |
| **Precious Metals** | XAUUSD, XAGUSD | Hohe Volatilität, Safe Haven |
| **US Indices** | US30, NAS100, US500.f | Trending Märkte, NY Session |
| **European Indices** | DE40, UK100, FR40 | Regionale Trends, EU Session |

### 🕒 **Session-optimierte Listen:**

#### **London Session (08:00-17:00 GMT):**
```
EURUSD
GBPUSD
EURGBP
DE40
UK100
```

#### **New York Session (13:00-22:00 GMT):**
```
EURUSD
XAUUSD
US30
NAS100
US500.f
```

#### **Asian Session (00:00-09:00 GMT):**
```
AUDUSD
NZDUSD
EURJPY
USDJPY
```

---
**💡 Tipp:** Starte immer mit `symbols_demo.txt` (nur EURUSD) für erste Tests!