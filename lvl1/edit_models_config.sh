cd ./models
perl -i -pe 's/window: \d+/window: 128/g' ./*
perl -i -pe 's/delay: \d+ \//delay: 256 \//g' ./*
perl -i -pe 's/skip: \d+ \//skip: 10 \//g' ./*

