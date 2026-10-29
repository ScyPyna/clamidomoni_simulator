"""Italian figure labels for the report scripts, switched on with FIG_LANG=it.

Importing this module does nothing unless the environment variable FIG_LANG is "it". When it is,
every matplotlib text is translated as it is set (titles, labels, legends, annotations), and
figures that would be saved under figs/en/ are saved under figs/it/ instead. The English strings
stay the single source in each script; the table below only maps them.

    FIG_LANG=it python3 report_figs.py
"""
import os
import re

if os.environ.get("FIG_LANG") == "it":
    import matplotlib.text
    import matplotlib.figure

    # whole sentences first, then fragments; applied in order
    RULES = [
        (r"Simulations: value at the end of the run, mean of 5 seeds\. Lab values read by eye from the lab figure, 5 acquisitions per point\.",
         "Simulazioni: valore a fine run, media di 5 seed. Valori di laboratorio letti a occhio dalla figura, 5 acquisizioni per punto."),
        (r"Above 1: 10 s of light accumulates more than 15 s\. Bars: standard error of the ratio of 5-seed means\. Crosses at the bottom: inside of the ring emptied \(rho_in/rho_out < 1\), ratio not shown\.",
         "Sopra 1: 10 s di luce accumulano più di 15 s. Barre: errore standard del rapporto tra medie su 5 seed. Croci in basso: interno dell'anello svuotato (rho_in/rho_out < 1), rapporto non mostrato."),
        (r"5 seeds pooled, last 1258 s of each run, 10 bins as in the lab pipeline\. Dotted line: isotropic, 1/π\. A state holding < 3 % of samples is not drawn\.",
         "5 seed uniti, ultimi 1258 s di ogni run, 10 bin come nel pipeline di laboratorio. Puntinata: isotropa, 1/π. Uno stato con meno del 3 % dei campioni non è disegnato."),
        (r"Thin lines: the 8 conditions \(τm 8/16 s, light 10/15 s, dark 1/5 s\); thick: median\. Positive = moving towards the light\. States holding < 3 % of samples are left out\.",
         "Linee sottili: le 8 condizioni (τm 8/16 s, luce 10/15 s, buio 1/5 s); spessa: mediana. Positivo = moto verso la luce. Esclusi gli stati con meno del 3 % dei campioni."),
        (r"White hatching: slope not distinguishable from zero at 2σ \(σ propagated from the spread over 5 seeds\)\. Zero = density blind to memory\.",
         "Tratteggio bianco: pendenza non distinguibile da zero a 2σ (σ propagato dalla dispersione tra i 5 seed). Zero = densità cieca alla memoria."),
        (r"Accumulation inside the ring \(r < (\d+) µm\) under pulsed light, 180 µW ring — (\d+) seeds per point",
         r"Accumulo dentro l'anello (r < \1 µm) a luce pulsata, anello 180 µW — \2 seed per punto"),
        (r"P\(θ\) computed as in the lab pipeline vs from the instantaneous direction",
         "P(θ) calcolata come nel pipeline di laboratorio e dalla direzione istantanea"),
        (r"P\(θ\) from 0\.51 s velocities, split by the reversal state",
         "P(θ) dalle velocità su 0.51 s, scomposta per stato di inversione"),
        (r"Averaging the velocity over longer windows sharpens both states, the total stays at zero",
         "Mediare la velocità su finestre più lunghe rende più netti entrambi gli stati, il totale resta a zero"),
        (r"Same light dose, different rhythm: long memories average the rhythm out",
         "Stessa dose di luce, ritmo diverso: le memorie lunghe cancellano il ritmo"),
        (r"labels at right: change from 2 s to 40 s period    \|    bars: seed-to-seed spread",
         "etichette a destra: variazione da periodo 2 s a 40 s    |    barre: dispersione tra seed"),
        (r"Lab and model cover the same range; the model misses the 10 s > 15 s ordering",
         "Laboratorio e modello coprono lo stesso intervallo; il modello non riproduce 10 s > 15 s"),
        (r"Low thresholds give the lab ordering \(10 s > 15 s\), but at 1 s dark they empty the ring",
         "Le soglie basse danno l'ordine del laboratorio (10 s > 15 s), ma a buio 1 s svuotano l'anello"),
        (r"A flat line means the density cannot tell the memories apart",
         "Una linea piatta significa che la densità non distingue le memorie"),
        (r"Where the density carries information about the memory",
         "Dove la densità porta informazione sulla memoria"),
        (r"the two states are polarised in opposite directions; the total stays at zero",
         "i due stati sono polarizzati in verso opposto; il totale resta a zero"),
        (r"Radial density shape: the lab accumulates in the centre of the ring",
         "Forma del profilo radiale: in laboratorio le alghe si accumulano al centro dell'anello"),
        (r"Grey band: the illuminated ring \(light above half its peak\)\. C = centre \(0-100 µm\) / ring \(150-250 µm\)\. Simulations at the working threshold 3\.0708, 5 seeds, light-on frames of the last 419 s\.",
         "Banda grigia: l'anello illuminato (luce sopra metà del picco). C = centro (0-100 µm) / anello (150-250 µm). Simulazioni alla soglia di lavoro 3.0708, 5 seed, fotogrammi a luce accesa degli ultimi 419 s."),
        (r"Simulations at the working threshold 3\.0708\. Dashed line: centre as dense as the ring\.",
         "Simulazioni alla soglia di lavoro 3.0708. Tratteggio: centro denso quanto l'anello."),
        (r"Above 1: more cells in the centre than on the ring", "Sopra 1: più cellule al centro che sull'anello"),
        (r"centre / ring density  \(0-100 µm / 150-250 µm\)", "densità centro / anello  (0-100 µm / 150-250 µm)"),
        (r"lab \(440 nm, end of last light phase\)", "lab (440 nm, fine dell'ultima fase di luce)"),
        (r"distance from the ring centre r \(µm\)", "distanza dal centro dell'anello r (µm)"),
        (r"ρ\(r\) / mean density within 600 µm", "ρ(r) / densità media entro 600 µm"),
        (r"no lab data", "nessun dato lab"),
        (r" split by the sign state of the gradient response", " scomposta per stato del segno della risposta al gradiente"),
        (r" by sign state — ", " per stato del segno — "),
        (r"threshold scan", "scan della soglia"),
        (r"dashed line: 1 = uniform density", "tratteggio: 1 = densità uniforme"),
        (r"seed-to-seed spread \(5 seeds\)", "dispersione tra i 5 seed"),
        (r"standard error of the seed mean", "errore standard sulla media dei seed"),
        (r"\bbars: ", "barre: "),
        (r"\(angle between motion and light gradient\)", "(angolo tra moto e gradiente di luce)"),
        (r"inside the ring", "dentro l'anello"),
        (r"velocity window \(s\); 0 = instantaneous direction", "finestra della velocità (s); 0 = direzione istantanea"),
        (r" lab\n window", " finestra\n lab"),
        (r"lab method: velocity over 0\.51 s", "metodo di laboratorio: velocità su 0.51 s"),
        (r"instantaneous swimming direction", "direzione di nuoto istantanea"),
        (r"U depth", "profondità U"),
        (r"cycle period \(s\), light on half of the time", "periodo del ciclo (s), luce accesa per metà del tempo"),
        (r"\(mean over last 419 s\)", "(media sugli ultimi 419 s)"),
        (r"lab \(read off figure\)", "lab (letto dalla figura)"),
        (r"dark duration \(s\)", "durata del buio (s)"),
        (r"light duration \(s\)", "durata della luce (s)"),
        (r"light on (\d+) s", r"luce accesa \1 s"),
        (r"light (\d+) s / dark", r"luce \1 s / buio"),
        (r"uniform = ", "uniforme = "),
        (r"current\nvalue", "valore\nattuale"),
        (r"sensitivity to memory", "sensibilità alla memoria"),
        (r"full range", "tutta la scala"),
        (r"closest pair", "coppia più vicina"),
        (r"fraction of samples in the reversed state \(%\)", "frazione di campioni in stato invertito (%)"),
        (r"positive = swimming towards the light", "positivo = nuota verso la luce"),
        (r"whole population", "popolazione totale"),
        (r"not reversed", "non invertite"),
        (r"\breversed\b", "invertite"),
        (r"\btotal\b", "totale"),
        (r"\bthreshold\b", "soglia"),
        (r"\bthr\b", "soglia"),
        (r"\bdark\b", "buio"),
    ]
    RULES = [(re.compile(a), b) for a, b in RULES]

    def translate(s):
        for rx, rep in RULES:
            s = rx.sub(rep, s)
        return s

    _set_text = matplotlib.text.Text.set_text

    def set_text(self, s):
        if isinstance(s, str) and s:
            s = translate(s)
        return _set_text(self, s)

    matplotlib.text.Text.set_text = set_text

    _savefig = matplotlib.figure.Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        f = str(fname)
        if "/figs/en/" in f:
            f = f.replace("/figs/en/", "/figs/it/")
            os.makedirs(os.path.dirname(f), exist_ok=True)
        return _savefig(self, f, *args, **kwargs)

    matplotlib.figure.Figure.savefig = savefig
