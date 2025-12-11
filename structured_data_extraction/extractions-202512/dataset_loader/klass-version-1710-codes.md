# Offisiell kommuneliste fra SSB

*Fra perplexity.ai:*

"Du kan hente kommunelista som CSV fra SSB på to enkle måter: via nettsiden (klikk «Last ned CSV») eller direkte via KLASS-API-et.

## 1. Manuell nedlasting i nettleser

1. Gå til SSB KLASS for kommuneinndeling:
   [https://www.ssb.no/klass/klassifikasjoner/131](https://www.ssb.no/klass/klassifikasjoner/131)[](https://www.ssb.no/klass/klassifikasjoner/131)
2. Velg riktig versjon (for dagens kommuner typisk «Kommuneinndeling 2024», gyldig fra 1.1.2024).[](https://www.ssb.no/klass/klassifikasjoner/131/korrespondanser/2472)
3. Inne på versjonen ser du en knapp/lenke «Last ned CSV», som gir deg komplett liste med kommunenummer og kommunenavn.[](https://www.ssb.no/klass/klassifikasjoner/131/versjon/1846)

## 2. Direkte via API (nyttig for skripting)

KLASS har et REST-API dokumentert her:
[https://data.ssb.no/api/klass/v1/api-guide.html](https://data.ssb.no/api/klass/v1/api-guide.html)[](https://data.norge.no/nb/datasets/15d63821-210d-4cdb-be62-927b3c7f1cb6/kommuneinndeling)

For kommuneinndeling bruker du klassifikasjon 131, og en konkret versjon-id. Eksempel (versjons-id er bare illustrasjon):

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-normal bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">bash</div></div><div><span><code><span><span class="token token">curl</span><span></span><span class="token token">"https://data.ssb.no/api/klass/v1/versions/3262.csv"</span><span> -o kommuneinndeling.csv
</span></span><span></span></code></span></div></div></div></pre>

Versjons-ID-en finner du ved å åpne ønsket versjon i nettleser; URL-en inneholder `/versjon/<id>`.[](https://www.ssb.no/klass/klassifikasjoner/131/versjon/3262)

For et mer robust opplegg kan du først slå opp gjeldende versjon for klassifikasjon 131 via API-et og så hente `.csv`-varianten av den.[](https://www.ssb.no/api)"
