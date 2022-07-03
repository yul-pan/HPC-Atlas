#!/use/bin/perl
#Position-speciﬁc scoring matrix (PSSM)
use warnings;

sub single{
    my @ID = @_;
    my %count;
    undef @single;
    my @single = grep {++$count{$_} < 2;} @ID;
    return @single;
}

sub POSSUM{
    my $name = @_;
    open(ONE,$_[0]);
    $line = 1;
    undef @PO;
    while(<ONE>){
        if($line == 1){
            $line++;
            next;
        }
        @PO = split(/\,/,$_);
    }
    close ONE;
    return @PO;
}


$File1 = "total_PPI.txt";  #comprehensive PIN 
open(ONE,$File1);
while(<ONE>){
    chomp;
    undef @data1;
    @data1 = split(/\s+/,$_);
    push(@Name,$data1[0]." ".$data1[1]);
}
close ONE;
@uniprotID = single(@Name);

$outfile = "New_possum.txt";  #Feature output file
open(OUT,">$outfile");
foreach $value1 (@uniprotID){
    @uniName = split(/\s+/,$value1);
    $File4 = "/Users/pyl/Code/Feature/POSSUM的副本/".$uniName[0].".txt";
    $File5 = "/Users/pyl/Code/Feature/POSSUM的副本/".$uniName[1].".txt";
    if(-e $File4 and -e $File5){
        @PO1 = POSSUM($File4);
        @PO2 = POSSUM($File5);
        $a = 0;
        $b = 0;
        print OUT $value1." ";
        foreach $key (0..@PO1-1){
            $a = ($PO1[$key] * $PO2[$key])**2 + $a;
            $b = $PO1[$key] + $PO2[$key];
            print OUT $b." ";
        }
        $similiar = sqrt($a);
        print OUT $similiar."\n";
    }else{
        print OUT $value1." ";
        $c = 1;
        while($c <= 421){
            print OUT "0 ";
            $c++;
        }
        print OUT "\n";
    }
}
close OUT;

