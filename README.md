# CaseStudie_Titanic | Überlebensvorhersage mit Logistischer Regression und Random Forest

Die Herausforderung ist, das Überleben anhand von Passagierdaten wie Alter, Geschlecht, Ticketklasse und Familiengröße vorherzusagen. 
Es wurde sich final für die logistische Regression entschieden, wegen der Performance bei binären Klassifizierungsaufgaben und Interpretierbarkeit.

## Datenbeschreibung

### Der Datensatz ist in drei Teile unterteilt:
- train.csv | für das Training des Modells
- test.csv | für die Modellvorhersagen
- gender_submission.csv | für IDs und tatsächlicher Überlebensstatus

### Merkmale umfassen:
* survival | Zielgröße
* pclass | Wahrscheinlich wurden Passagiere 1er Klasse Vorrang beim Besetzen von Booten gelassen - Analog 3. Klasse als letztes gerettet.
* sex | Besondern junge/alte Menschen und teilweise Frauen haben es im direkten Kampf zu den Rettungsbooten wahrscheinlich schwerer gehabt. Jedoch, wurde wahrscheinlich Kindern, Alten und Frauen Vorrang beim Besetzen von Booten gelassen.
* Age | Besondern junge/alte Menschen und teilweise Frauen haben es im direkten Kampf zu den Rettungsbooten wahrscheinlich schwerer gehabt. Jedoch, wurde wahrscheinlich Kindern, Alten und Frauen Vorrang beim Besetzen von Booten gelassen.
* sibsp | Mehr Familienmitglieder veringern die "Jeder für sich" Mentalität, die zum frühzeitigen Erreichen eines Rettungsbootes wichtig ist. Mit Kindern/Eltern wird im Chaos des Unfalls wahrscheinlich nicht nur als sich gedacht, was die Überlebenschancen verringern könnte.
* parch | Mehr Familienmitglieder veringern die "Jeder für sich" Mentalität, die zum frühzeitigen Erreichen eines Rettungsbootes wichtig ist. Mit Kindern/Eltern wird im Chaos des Unfalls wahrscheinlich nicht nur als sich gedacht, was die Überlebenschancen verringern könnte.
* ticket | Wahrscheinlich nicht relevant
* fare | Information zum Einkommen/Sozialen Rang (ähnlich wie pclass)
* cabin | KabinenNr -> Möglicherweise Information zur Position des Passagieres an Bord während des Unfalls. Ebenfalls Information zum Einkommen/Sozialen Rang (ähnlich wie pclass)
* embarked | Hafen des an Bord gehens:	C = Cherbourg, Q = Queenstown, S = Southampton

## Methodik

1. **Datenexploration**: Erste Datenanalyse um Merkmalsverteilungen und -beziehungen zu verstehen.
2. **Feature Engineering**: Verbesserung des Modells mit neuen, aus bestehenden Daten abgeleiteten Merkmalen.
3. **Datenvorverarbeitung**: Bereinigung der Daten durch Behandlung fehlender Werte, Encoding kategorischer Variablen und Skalierung der numerischen Features.
4. **Modellerstellung**: Implementierung der Modelle zur Vorhersage des Überlebens.
5. **Modellbewertung**: Beurteilung der Modellleistung anhand von Performance und anderen Metriken.

## Ergebnisse

Das Modell der logistischen Regression erreichte eine Genauigkeit von über 90% auf dem Testset. Detaillierte Informationen und die Konfusionsmatrix werden in den Projekt-Notebooks diskutiert.
Informtaioen u.a. zur FeatureImportance und Vorhersage können ebenfalls den Plots und Output_Data entnommen werden.

## Installation

```bash
git clone https://github.com/ATE91/CaseStudie_Titanic.git

pip install -r requirements.txt

