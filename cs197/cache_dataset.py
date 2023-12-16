from datasets import load_dataset

def main():
    #uspto_train_dataset = load_dataset('allyc/My-Dataset', 'uspto', split='train', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/uspto/train').with_format('torch')
    #print(uspto_train_dataset[0])
    #uspto_validation_dataset =  load_dataset('suolyer/pile_uspto', split='validation', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/uspto/validation').with_format('torch')
    #print(uspto_validation_dataset[0])
    #pubmed_train_dataset =  load_dataset('allyc/My-Dataset','pubmed', split='train', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/pubmed/train').with_format('torch')
    #print(pubmed_train_dataset[0])
    # pubmed_validation_dataset =  load_dataset('suolyer/pile_pubmed-abstracts', split='validation', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/pubmed/validation').with_format('torch')
    # print(pubmed_validation_dataset[0])
    # openwebtext2_val = load_dataset('suolyer/pile_openwebtext2', split='validation', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/openwebtext/validation').with_format('torch')  
    # pile_uncoprighted_val = load_dataset('monology/pile-uncopyrighted', split='validation', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/pile-uncopyrighted/validation').with_format('torch')  
    # slim_pajama_val = load_dataset('venketh/SlimPajama-62B', split='validation', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/slim-pajama/validation').with_format('torch') 
    pile_val = load_dataset('allyc/My-Dataset', 'all', split='validation[:1000]', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/pile/validation').with_format('torch')
 



if __name__ == '__main__':
    main()
