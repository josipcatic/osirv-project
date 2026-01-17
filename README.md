# Nelinearna registracija fotografija lica na prosječno lice pomoću ključnih točaka

## Dependencies 

Ovaj projekt koristi Openface za nelinearnu registraciju lica.

OpenFace NIJE uključen u ovaj repository.
Molim Vas instalirajte OpenFace sa:
https://github.com/TadasBaltrusaitis/OpenFace
potrebno je postaviti openface u root mapu kao ./openface/

Također potrebno je instalirati .dat datoteke sa:
https://www.dropbox.com/scl/fo/pq55xsw9eabf346vivmqn/AClMzt769mNe8ISrPjL9Bdo?rlkey=7qq9uk66x877ck4nny45qdzn2&e=3&dl=0
potrebne .dat datoteke su: cen_patches_0.25_of.dat, cen_patches_0.35_of.dat, cen_patches_0.5_of.dat, cen_patches_0.1_of.dat.
.dat datoteke potrebno je staviti u ./openface/model/patch_experts/