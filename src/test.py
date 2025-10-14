from retriver import CPIRetriever

r = CPIRetriever()
results = r.search("wheat prices in Lahore in 2020", k=3)
for res in results:
    print(res, "\n")
