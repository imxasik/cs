"""
Ports / coastal places used on the map and in the port table.

Everything is plain data: ``name -> (latitude, longitude)``. The dict is
ordered roughly along the coastline (Bangladesh -> India east coast ->
Sri Lanka -> Myanmar -> Thailand/Malaysia -> Sumatra) which also keeps the
"which side of the landfall" split easy to eyeball.

Add, remove or rename entries as you like — nothing else needs editing.
Coordinates are approximate (city / port centre) and are meant for map
annotation and distance estimates, not for navigation.
"""


def BOB():
    return {
        # ---------------- Bangladesh ----------------
        "Barishal": (22.7000, 90.3700),
        "Bhola": (22.6900, 90.6400),
        "Mongla": (22.4880, 89.5950),
        "Payra": (21.9850, 90.2860),
        "CTG": (22.3130, 91.8000),
        "CoxBazar": (21.4140, 91.9830),
        "Teknaf": (20.8700, 92.3000),

        # ---------------- India: West Bengal -> Odisha ----------------
        "Kolkata": (22.5700, 88.3600),
        "Haldia": (22.0600, 88.0600),
        "Sagar Is.": (21.6500, 88.0800),
        "Digha": (21.6258, 87.5156),
        "Contai": (21.7792, 87.7483),
        "Balasore": (21.4946, 86.9317),
        "Chandbali": (20.7800, 86.7400),
        "Paradip": (20.3160, 86.6110),
        "Puri": (19.8135, 85.8312),
        "Gopalpur": (19.2700, 84.9100),

        # ---------------- India: Andhra Pradesh ----------------
        "Srikakulam": (18.3000, 83.9000),
        "V.patnam": (17.7041, 83.2977),
        "Kakinada": (16.5758, 82.1518),
        "M.C.patnam": (16.1809, 81.1303),
        "Krishnapatnam": (14.2800, 80.1200),

        # ---------------- India: Tamil Nadu ----------------
        "Chennai": (13.0827, 80.2707),
        "Puducherry": (11.9300, 79.8300),
        "Cuddalore": (11.7500, 79.7500),
        "Nagapattinam": (10.7600, 79.8400),
        "Rameswaram": (9.2900, 79.3100),
        "Thoothukudi": (8.8000, 78.1300),
        "Kanyakumari": (8.0800, 77.5400),

        # ---------------- Sri Lanka ----------------
        "Jaffna": (9.6600, 80.0100),
        "Trincomalee": (8.5700, 81.2300),
        "Batticaloa": (7.7200, 81.7000),
        "Colombo": (6.9500, 79.8400),
        "Galle": (6.0300, 80.2200),
        "Hambantona": (6.1190, 81.1080),

        # ---------------- Myanmar ----------------
        "Sittwe": (20.1500, 92.9000),
        "Kyaukpyu": (19.4300, 93.5500),
        "Thandwe": (18.4700, 94.3700),
        "Pathein": (16.7800, 94.7300),
        "Yangon": (16.8000, 96.1500),
        "Mawlamyine": (16.4900, 97.6300),
        "Dawei": (14.0800, 98.2000),

        # ---------------- Thailand / Malaysia / Singapore ----------------
        "Phuket": (7.8800, 98.3900),
        "Penang": (5.4100, 100.3300),
        "Port Klang": (3.0000, 101.4000),
        "Singapore": (1.2900, 103.8500),

        # ---------------- Andaman & Nicobar ----------------
        "Port Blair": (11.6200, 92.7500),

        # ---------------- Indonesia (Sumatra) ----------------
        "Banda Aceh": (5.5500, 95.3200),
        "Meulaboh": (4.1400, 96.1300),
        "Belawan": (3.7167, 98.7167),
        "Tanjungbalai": (2.9667, 99.8000),
        "Sibolga": (1.7400, 98.7800),
        "Padang": (-0.9500, 100.3500),
        "Bengkulu": (-3.8000, 102.2600),
    }
