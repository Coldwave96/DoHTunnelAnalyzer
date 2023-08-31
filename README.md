# DoHTunnelAnalyzer

Domain Name System (DNS) is one of the early and vulnerable network protocols which has several security loopholes that have been exploited repeatedly over the years. To overcome some of the DNS vulnerabilities related to privacy and data manipulation, IETF introduced DNS over HTTPS (DoH) in RFC8484, a protocol that enhances privacy and combats eavesdropping and man-in-the-middle attacks by encrypting DNS queries and sending them in a covert channel/tunnel so that data is not hampered on the way.

This repo contructs an easy way to detect malicious DoH tunnels, which created by some DNS tunneling tools such as [dns2tcp](https://gitlab.com/kalilinux/packages/dns2tcp), [DNSCat2](https://github.com/iagox86/dnscat2), and [iodine](https://code.kryo.se/iodine).

## Datasets Details

The dataset used for this research is the CIRA-CIC-DoHBrw-2020 dataset developed by the Canadian Institute of Cybersecurity. This dataset can be found [here](https://www.unb.ca/cic/datasets/dohbrw-2020.html).

## Model Design
### Layer 1
* Random Forest Classifier (RF)
* Decision Tree Classifier (DT)

### Layer 2
* MLP (Multilayer Perceptron)

## Evaluation
||precision|recall|f1-score|support|
|-|-|-|-|-|
|benign|0.99|1.00|1.00|3905|
|dns2tcp|1.00|0.99|0.99|33577|
|DNSCat2|0.97|0.99|0.98|7106|
|iodine|0.98|0.99|0.98|9263|
||||||
|accuracy|||0.99|53851|
|macro avg|0.98|0.99|0.99|53851|
|weighted avg|0.99|0.99|0.99|53851|
