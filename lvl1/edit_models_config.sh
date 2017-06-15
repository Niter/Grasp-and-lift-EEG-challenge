cd ./models
perl -i -pe 's/window: \d+/window: 256/g' ./*
perl -i -pe 's/delay: \d+ \//delay: 512\//g' ./*
perl -i -pe 's/skip: \d+ \//skip: 20 \//g' ./*

