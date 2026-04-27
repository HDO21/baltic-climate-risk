# baltic-climate-risk

## Projekti eesmärk

Selle projekti eesmärk on luua Baltikumi (Eesti, Läti, Leedu) kliimaindikaatorite andmebaas, mis võimaldab hinnata kliimariske asukohapõhiselt ja pärida tulemusi lihtsal kujul töölaualt (nt Streamlit).

Andmebaas ühendab:
- **ajaloolise kliimaperioodi** andmed (1990–2020), peamiselt **ERA5-Land** allikast;
- **tulevikuprojektsioonid** kuni aastani 2100, peamiselt **EURO-CORDEX** andmestikust;
- peamised kliimamuutujad nagu **päevane miinimum- ja maksimumtemperatuur**, **sademed** ning võimalusel ka **tuul**.

Nende põhjal arvutatakse kliimaindikaatorid, näiteks:
- kuumapäevade arv,
- külmumis-sulamispäevade arv,
- troopilised ööd,
- äärmuslike sademetega päevad,
- järjestikused kuivad päevad.

Lõpptulemus on päringupõhine andmebaas, kust saab asukoha, perioodi ja stsenaariumi alusel tagastada vastava kliimaindikaatori väärtuse. Eesmärk on teha Baltikumi jaoks detailsem ja praktilisem lahendus kui Copernicus Climate Atlas, kasutades peenemat ruumilist ja ajalist resolutsiooni ning kohandatud indikaatoreid.

Töölaua prototüüp: https://baltic-climate-risk-2utxkc8rfxws4fktvmoy5g.streamlit.app/
