import random
import numpy as np
import requests
from math import radians, cos, sin, asin, sqrt
import folium


# --- PARAMETRY ---
LICZBA_KURIEROW = 3            # liczba kurierów
ROZMIAR_POPULACJI = 6000       # rozmiar populacji
LICZBA_POKOLEN = 200           # liczba pokoleń
P_MUTACJI = 0.2                # prawdopodobieństwo mutacji
TURNIEJ_SIZE = 5               # liczba uczestników turnieju
ALPHA_KARA = 0.02              # przyrost kary (alpha)
MAX_CZAS = 480                 # maksymalny czas pracy kuriera (min)
CZAS_POSTOJU = 20              # czas postoju kuriera przy paczkomacie (min)


# --- HAVERSINE ---
def haversine(lon1, lat1, lon2, lat2):
    """
    Przeliczenie wspolrzednych geograficznych paczkomatow na odleglosci miedzy nimi
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

# --- API ---
def API_paczkomaty():
    """
    Pobranie listy paczkomatow z API i uzycie funkcji haversine
    """
    url = "https://api-shipx-pl.easypack24.net/v1/points"
    params = {"per_page": 10000, "type": "parcel_locker", "city": "Lubin"}
    response = requests.get(url, params=params)
    points = response.json()

    lokalizacje = [(51.39936510651588, 16.205476630389562)]  # sortownia
    if 'items' in points:
        for item in points['items']:
            lokalizacje.append((item['location']['latitude'], item['location']['longitude']))

    n = len(lokalizacje)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = haversine(lokalizacje[i][1], lokalizacje[i][0], lokalizacje[j][1], lokalizacje[j][0])
    return dist_matrix, n, lokalizacje


# --- CZAS TRASY ---
def czas_trasy(podtrasa, dist_matrix):
    """
    Przeliczenie czasu jazdy kuriera
    :param podtrasa: trasa jednego kuriera
    :param dist_matrix: macierz odleglosci miedzy paczkomatami
    :return: calkowity czas pracy kuriera
    """
    dystans = dist_matrix[0][podtrasa[0]] if podtrasa else 0
    for i in range(len(podtrasa) - 1):
        dystans += dist_matrix[podtrasa[i]][podtrasa[i + 1]]
    dystans += dist_matrix[podtrasa[-1]][0] if podtrasa else 0
    czas_jazdy = dystans / 40 * 60  # 40km/h srednia predkosc
    czas_postoju = CZAS_POSTOJU * len(podtrasa)
    return czas_jazdy + czas_postoju

# --- INICJALIZACJA ---
def inicjalizuj_populacje(rozmiar_populacji, liczba_paczkomatow, liczba_kurierow):
    """
    Tworzenie startowej populacji
    """
    populacja = []
    for _ in range(rozmiar_populacji):
        paczkomaty = list(range(1, liczba_paczkomatow))
        random.shuffle(paczkomaty)
        podzialy = sorted(random.sample(range(1, len(paczkomaty)), liczba_kurierow - 1))
        osobnik = []
        last = 0
        for p in podzialy:
            osobnik += paczkomaty[last:p] + ['|']
            last = p
        osobnik += paczkomaty[last:]
        populacja.append(osobnik)
    return populacja

# --- KOSZT + KARA ---
def koszt_trasy(osobnik, dist_matrix, pokolenie=0, alpha=ALPHA_KARA):
    """
    Obliczenie kosztu trasy uwzgledniajac kary za przekroczenie czasu
    :param osobnik: trasa wszystkich kurierow w formacie [1, 2, 3, |, 4, 5, 6]. "|" to zmiana kuriera
    :param dist_matrix: macierz odleglosci miedzy paczkomatami
    :param pokolenie: ktora to populacja w iteracji
    :param alpha: szybkosc wzrostu funkcji kar
    :return: koszt trasy
    """
    laczny_koszt = 0
    laczna_kara = 0
    kara = (1 + alpha)**pokolenie   # funkcja kary
    podtrasa = []

    for gen in osobnik + ['|']:
        if gen == '|':
            if podtrasa:
                koszt = dist_matrix[0][podtrasa[0]]
                for i in range(len(podtrasa) - 1):
                    koszt += dist_matrix[podtrasa[i]][podtrasa[i + 1]]
                koszt += dist_matrix[podtrasa[-1]][0]
                czas = czas_trasy(podtrasa, dist_matrix)
                kara_kuriera = kara if czas > MAX_CZAS else 0
                laczny_koszt += koszt
                laczna_kara += kara_kuriera
                podtrasa = []
        else:
            podtrasa.append(gen)

    return laczny_koszt + laczna_kara, laczny_koszt, laczna_kara

# --- SELEKCJA ---
def selekcja(populacja, oceny, turniej=TURNIEJ_SIZE):
    """
    Turniejowy wybor najlepszego osobnika w populacji
    :param oceny: koszta osobnika
    :param turniej: ilu osobnikow bierze udzial w turnieju
    :return:
    """
    uczestnicy = random.sample(list(zip(populacja, oceny)), turniej)
    uczestnicy.sort(key=lambda x: x[1][0])
    return uczestnicy[0][0]

# --- KRZYŻOWANIE I MUTACJA ---
def crossover(p1, p2):
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))
    dziecko = [None] * size
    dziecko[start:end] = p1[start:end]
    p2_filtered = [item for item in p2 if item not in dziecko]
    idx = 0
    for i in range(size):
        if dziecko[i] is None:
            dziecko[i] = p2_filtered[idx]
            idx += 1
    return dziecko

def crossover_wiele_kurierow(p1, p2, liczba_kurierow):
    paczkomaty1 = [x for x in p1 if x != '|']
    paczkomaty2 = [x for x in p2 if x != '|']
    dziecko_paczkomaty = crossover(paczkomaty1, paczkomaty2)
    podzialy = sorted(random.sample(range(1, len(dziecko_paczkomaty)), liczba_kurierow - 1))
    dziecko = []
    last = 0
    for p in podzialy:
        dziecko += dziecko_paczkomaty[last:p] + ['|']
        last = p
    dziecko += dziecko_paczkomaty[last:]
    return dziecko

def mutacja(osobnik, prawdopodobienstwo=P_MUTACJI):
    if random.random() < prawdopodobienstwo:
        indeksy = [i for i, x in enumerate(osobnik) if x != '|']
        i, j = random.sample(indeksy, 2)
        osobnik[i], osobnik[j] = osobnik[j], osobnik[i]
    return osobnik

# --- ALGORYTM ---
def algorytm_genetyczny(dist_matrix, liczba_pokolen=LICZBA_POKOLEN, rozmiar_populacji=ROZMIAR_POPULACJI, liczba_kurierow=LICZBA_KURIEROW):
    populacja = inicjalizuj_populacje(rozmiar_populacji, len(dist_matrix), liczba_kurierow)

    for pokolenie in range(liczba_pokolen):
        oceny = [koszt_trasy(o, dist_matrix, pokolenie) for o in populacja]

        bezkarnosc = [(i, oceny[i][0]) for i in range(len(oceny)) if oceny[i][2] == 0]
        if bezkarnosc:
            idx_bezkarny = min(bezkarnosc, key=lambda x: x[1])[0]
            best_bezkarny = populacja[idx_bezkarny]
        else:
            best_bezkarny = None

        # Raportowanie najlepszego w generacji
        idx_best = int(np.argmin([o[0] for o in oceny]))
        best_bez_kary = oceny[idx_best][1]
        best_kara = oceny[idx_best][2]
        best_calkowity = oceny[idx_best][0]
        print(f"Pokolenie {pokolenie+1}: {best_bez_kary:.2f} + kara {best_kara:.2f} => {best_calkowity:.2f}")

        # Selekcja
        nowa_populacja = []
        while len(nowa_populacja) < rozmiar_populacji:
            r1 = selekcja(populacja, oceny)
            r2 = selekcja(populacja, oceny)
            d1 = crossover_wiele_kurierow(r1, r2, liczba_kurierow)
            d2 = crossover_wiele_kurierow(r2, r1, liczba_kurierow)
            nowa_populacja.append(mutacja(d1))
            nowa_populacja.append(mutacja(d2))

        if best_bezkarny is not None:
            oceny_nowe = [koszt_trasy(o, dist_matrix, pokolenie) for o in nowa_populacja]
            worst_idx = int(np.argmax([o[0] for o in oceny_nowe]))
            nowa_populacja[worst_idx] = best_bezkarny

        populacja = nowa_populacja

    # Finalna ewaluacja
    oceny = [koszt_trasy(o, dist_matrix, liczba_pokolen - 1) for o in populacja]
    idx_best = int(np.argmin([o[0] for o in oceny]))
    najlepszy_osobnik = populacja[idx_best]
    koszt_razem, koszt_bez_kary, kara = oceny[idx_best]
    return najlepszy_osobnik, koszt_razem, koszt_bez_kary, kara


# --- MAPA ---
def pokaz_trase_na_mapie(osobnik, lokalizacje, nazwa_pliku="trasa.html"):
    """
    Zapisanie mapy z trasami do pliku
    """
    podzialy = []
    trasa = []
    for x in osobnik + ['|']:
        if x == '|':
            if trasa:
                podzialy.append(trasa)
                trasa = []
        else:
            trasa.append(x)

    mapa = folium.Map(location=lokalizacje[0], zoom_start=12)
    folium.Marker(location=lokalizacje[0], popup="Sortownia", icon=folium.Icon(color='black')).add_to(mapa)

    kolory = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'lightblue']
    for i, kurier_trasa in enumerate(podzialy):
        kolor = kolory[i % len(kolory)]
        pelna = [lokalizacje[0]] + [lokalizacje[j] for j in kurier_trasa] + [lokalizacje[0]]
        folium.PolyLine(pelna, color=kolor, weight=3).add_to(mapa)
        for idx, pkt in enumerate(kurier_trasa):
            folium.Marker(location=lokalizacje[pkt], popup=f"Kurier {i + 1} - {idx + 1}",
                          icon=folium.Icon(color=kolor)).add_to(mapa)

    mapa.save(nazwa_pliku)
    print(f"Mapa zapisana jako {nazwa_pliku}")


if __name__ == "__main__":
    dist_matrix, n, lokalizacje = API_paczkomaty()
    najlepszy, koszt_laczny, koszt_bez_kary, kara = algorytm_genetyczny(dist_matrix, liczba_kurierow=LICZBA_KURIEROW)
    print("\nNajlepsza trasa:", najlepszy)
    print(f"Koszt (bez kary): {koszt_bez_kary:.2f}")
    print(f"Kara: {kara:.2f}")
    print(f"Łączny koszt: {koszt_laczny:.2f}")
    pokaz_trase_na_mapie(najlepszy, lokalizacje)