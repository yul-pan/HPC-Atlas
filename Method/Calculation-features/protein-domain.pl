#!/use/bin/perl
#Protein domain interactions.
use warnings;

sub single{
    my @ID = @_;
    my %count;
    undef @single;
    my @single = grep {++$count{$_} < 2;} @ID;
    return @single;
}

$File1 = "total_PPI.txt";  #comprehensive PIN ;
open(ONE,$File1);
while(<ONE>){
    chomp;
    undef @data1;
    @data1 = split(/\s+/,$_);
    push(@Name,$data1[0]." ".$data1[1]);
}
close ONE;
@uniprotID = single(@Name);

###extract domain pairs
$File5 = "pfamA_interactions.txt";
open(TWO,$File5);
while(<TWO>){
    chomp;
    push(@Domain,$_);
}
close TWO;
@Domain = single(@Domain);

###Calculate features
$File4 = "/Users/pyl/Code/Feature/Protein_Domain/New_Pro-domain.txt";
$outFile = "/Users/pyl/Code/Feature/Protein_Domain/New_PD_feature.txt";
open(OUT,">$outFile");
foreach $value1 (@uniprotID) {
    undef @proteinName;
    undef @protein1;
    undef @protein2;
    @proteinName = split(/\s+/,$value1);
    open(TWO,$File4);
    while(<TWO>){
        chomp;
        undef @data2;
        @data2 = split(/\s+/,$_);
        if(scalar(@data2) > 1){
            if($proteinName[0] eq $data2[0]){
                @protein1 = @data2;
            }
            if($proteinName[1] eq $data2[0]){
                @protein2 = @data2;
            }
        }           
    }
    close TWO;

    if(scalar(@protein1) > 1 and scalar(@protein2) > 1){
        $inter = 0;
        $same = 0;
        foreach $key1 (1..@protein1-1){
            foreach $key2 (1..@protein2-1){
                if($protein1[$key1] eq $protein2[$key2]){
                    $same++;
                }
                foreach $value2 (@Domain){
                    @Pfam = split(/\s+/,$value2);
                    if(($protein1[$key1] eq $Pfam[0] and $protein2[$key2] eq $Pfam[1]) or ($protein1[$key1] eq $Pfam[1] and $protein2[$key2] eq $Pfam[0])){
                        $inter++;
                        last;
                    }
                }
            }
        }

        $total = scalar(@protein1) + scalar(@protein2) - 2;
        $sim_same = $same/$total;
        $sim_inter = $inter/$total;
        $feature = $same." ".$inter." ".$total." ".$sim_same." ".$sim_inter;
    }else{
        $total = scalar(@protein1) + scalar(@protein2) - 2;
        $feature = "0 0 ".$total." 0 0";
    }

    print OUT $proteinName[0]." ".$proteinName[1]." ".$feature."\n";
}
close OUT;