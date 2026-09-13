# Land Use Frontier Engine

A multi objective spatial optimization laboratory for an agricultural frontier, built by Yinka Aderibigbe as independent candidate work product for the doctoral position Spatial Optimization of Land Use in Brazil (project Frontiers of MATOPIBA landscapes) at Utrecht University, Department of Human Geography and Spatial Planning. Everything is synthetic and illustrative; no real parcel, person or municipality is represented.

## What it does

1. Generates a synthetic frontier landscape in the spirit of the MATOPIBA region: existing soy on the flat high aptitude plateau, pasture and degraded pasture below it, cerrado woodland and grassland in between, gallery forest along the drainage, protected blocks, recharge zones on the sandy plateau, and an endemism field. The decision space is every native cell in the expansion band (keep or convert to soy) and every degraded pasture cell (keep, intensify to soy, or restore toward native cover); riparian cells can never become cropland, and a repair operator keeps native cover above a Forest Code style floor of 35 percent of unprotected land in every plan.
2. Scores every plan on four indicators: production, carbon stock, biodiversity (habitat value weighted by endemism plus a connectivity term), and water services (plateau recharge infiltration and gallery forest integrity). Yields carry two landscape feedbacks: a local adjacency benefit from neighbouring native vegetation, and a regional rainfall recycling penalty that grows with total clearing, following the evidence that clearing the cerrado dries its own agriculture.
3. Computes the Pareto frontier with an NSGA-II style evolutionary search (fast nondominated sorting, crowding distance, uniform crossover, per gene mutation), warm started with disclosed reference plans: the current landscape, trend expansion, degraded pasture first, full restoration, aptitude ranked expansion at six densities, two water sensitive variants, and the 13 weighted overlay plans, so the returned frontier can only match or improve on the practice it is compared against.
4. Compares three ways of planning the same expansion: trend expansion (aptitude first over native cerrado, the historical pattern), weighted overlay ranking (the common one pass GIS practice), and the evolutionary frontier, at matched production.
5. Recomputes the frontier under a drier 2040s climate and reads it against a fixed demand, in one shared currency (today's landscape under today's climate), so scenario differences are visible instead of being normalized away.
6. Translates stakeholder preferences into impact indicators and back into plans: four panels weigh the indicators differently and land on different frontier plans, and a basin committee's stated concern for water is translated two defensible ways (plateau recharge, gallery forest integrity) that select visibly different maps, which is the engine's core argument for doing indicator translation in the field work rather than in the code alone.

All numbers and claims on the page are recomputed live from the run configured in the sidebar; nothing is hard coded.

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

## Testing

`stub_test.py` in the repository root executes the full page headlessly with a stubbed Streamlit, prints every caption, and checks defined pass markers (frontier size, matched plan beats trend expansion on every environmental objective at equal or higher production, overlay domination and coverage, drier climate lowers attainable production, translation ambiguity changes the selected map, native cover floor holds). The markers pass on seeds 1 through 7.

## References

Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6(2), 182 to 197.
Kaim, A., Cord, A. F. and Volk, M. (2018). A review of multi-criteria optimization techniques for agricultural land use allocation. Environmental Modelling & Software 105, 79 to 93.
Verstegen, J. A., van der Hilst, F., Woltjer, G., Karssenberg, D., de Jong, S. M. and Faaij, A. P. C. (2016). What can and can't we say about indirect land-use change in Brazil using an integrated economic - land-use change model? GCB Bioenergy 8(3), 561 to 578.
Verstegen, J. A., Karssenberg, D., van der Hilst, F. and Faaij, A. (2012). Spatio-temporal uncertainty in Spatial Decision Support Systems: a case study of changing land availability for bioenergy crops in Mozambique. Computers, Environment and Urban Systems 36(1), 30 to 42.
Hildemann, M., Pebesma, E. and Verstegen, J. A. (2023). Multi-objective Allocation Optimization of Soil Conservation Measures Under Data Uncertainty. Environmental Management 72(5), 959 to 977.
Strassburg, B. B. N. and colleagues (2017). Moment of truth for the Cerrado hotspot. Nature Ecology & Evolution 1, 0099.
Soterroni, A. C. and colleagues (2019). Expanding the Soy Moratorium to Brazil's Cerrado. Science Advances 5(7), eaav7336.
Spera, S. A., Galford, G. L., Coe, M. T., Macedo, M. N. and Mustard, J. F. (2016). Land-use change affects water recycling in Brazil's last agricultural frontier. Global Change Biology 22(10), 3405 to 3413.

## Licence

MIT. Author: Yinka Aderibigbe, ORCID 0009-0000-1726-5564.
