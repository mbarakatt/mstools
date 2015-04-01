ms 2000 1 -t 13.0 -I 3 1000 0 1000 1.0 -r 100.0 2500  | tail -2000 | python collapse_haplotype.py > haps1.txt 
ms 2000 1 -t 13.0 -I 7 1000 0 0 0 0 0 1000 1.0 -r 100.0 2500  | tail -2000 | python collapse_haplotype.py > haps2.txt 

python pca.py haps2.txt & python pca.py haps1.txt

open haps1.txt.jpg haps2.txt.jpg

