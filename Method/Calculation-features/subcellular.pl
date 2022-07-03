#!/use/bin/perl
#Subcellular localization.
use warnings;
use LWP::Simple;

sub single{
    my @ID = @_;
    my %count;
    undef @single;
    my @single = grep {++$count{$_} < 2;} @ID;
    return @single;
}


$Fil1 = "BioPlex_PPI.txt";
$File2 = "HuRI_PPI.txt";
$File3 = "other_PPI.txt";
open(ONE,$Fil1);
while(<ONE>){
    chomp;
    undef @data1;
    @data1 = split(/\s+/,$_);
    push(@Name,$data1[2]);
    push(@Name,$data1[3]);
}
close ONE;

open(ONE,$File2);
while(<ONE>){
    chomp;
    undef @data1;
    @data1 = split(/\s+/,$_);
    push(@Name,$data1[2]);
    push(@Name,$data1[3]);
}
close ONE;

open(ONE,$File3);
while(<ONE>){
    chomp;
    undef @data1;
    @data1 = split(/\s+/,$_);
    push(@Name,$data1[4]);
    push(@Name,$data1[5]);
}
close ONE;

@uniprotID = single(@Name);

$SwissPortFile = "SwissPort_subcell.txt";
$OutFile = "subID_2.txt";
open(OUT,">$OutFile");

foreach $value (@uniprotID){
    ###download xml file
    $url = "https://www.uniprot.org/uniprot/".$value.".xml";
    $HTMLFile = "uniport.xml";
    getstore ($url, $HTMLFile);
    open(TWO,$HTMLFile);
    $switch = 0;
    undef @subName;
    while(<TWO>){
        chomp;
        undef @data2;
        undef @data3;
        if($_=~/^\<subcellularLocation/){
            $switch = 1;
            next;
        }
        if($_=~/^\<\/subcellularLocation/){
            $switch = 0;
            next;
        }
        if($switch == 1){
            @data2 = split(/\>/,$_);
            @data3 = split(/\</,$data2[1]);
            push(@subName,$data3[0]);
        }
    }
    close TWO;
    
    ####Mapping the subcellular name in the previous step to the subcellular ID
    open(TWO,$SwissPortFile);
    $switch = 0;
    $switchGO = "OFF";
    undef @subID;
    while(<TWO>){
        chomp;
        if($_=~/^ID/ or $_=~/^IO/ or $_=~/^IT/){
            undef @data4;
            @data4 = split(/   /,$_);
            foreach $a (@subName){
                $data4[1]=~s/\.$//;
                if($data4[1] eq $a){
                    $switchGO = "ON";
                    last;
                }
            }
        }
        
        if($switchGO eq "ON" and $_=~/^AC/){
            undef @data5;
            undef @temp;
            @data5 = split(/   /,$_);
            push(@temp,$data5[1]);
            $switch = 1;
        }
       
        if($_=~/^\/\//){
            $switchGO = "OFF";
            next;
        }
        if($switchGO eq "ON" and $_=~/^GO/){
            undef @data6;
            @data6 = split(/\s+/,$_);
            $data6[1] =~s/\;$//;
            push(@temp,$data6[1]);
        }
        if($switchGO eq "OFF" and $switch == 1){
            if(scalar(@temp) == 2){
                push(@subID,$temp[1]);
            }else{
                push(@subID,$temp[0]);
            }
            $switch = 0;
        }
    }
    close TWO;


    $url = "https://www.uniprot.org/uniprot/".$value.".txt";
    $HTMLFile = "uniport.txt";
    getstore ($url, $HTMLFile);
    open(THREE,$HTMLFile);
    while(<THREE>){
        chomp;
        if($_=~/^DR   GO\; GO\:/){
            undef @data2;
            @data2 = split(/\s+/,$_);
            if($data2[3]=~/^C\:/){
                $data2[2] =~s/\;$//;
                push(@subID,$data2[2]);
            }
        }
    }
    close THREE;
    @subID = single(@subID);

    $endValue = "";
    foreach $b (@subID){
        $endValue .= $b." ";
    }
    print OUT $value." ".$endValue."\n";
    print $value." ";
}
close OUT;
