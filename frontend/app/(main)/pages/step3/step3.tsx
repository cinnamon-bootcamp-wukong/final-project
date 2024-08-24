'use client';
import { Button } from 'primereact/button';
import { Tooltip } from 'primereact/tooltip';
import { useState, useEffect } from 'react';
import { MenuItem } from 'primereact/menuitem';
import { Steps } from 'primereact/steps';
import { useRouter } from 'next/navigation';
import { ProgressSpinner } from 'primereact/progressspinner';
import { Image } from 'primereact/image';

interface step3Prob {
    setStep: React.Dispatch<React.SetStateAction<number>>;
    pass2Step: boolean;
}

const Step3: React.FC<step3Prob> = ({setStep, pass2Step}) => {
    const [loading, setLoading] = useState(true);
    const [file, setFile] = useState<File | null>(null);
    const [images, setImages] = useState<string[]>([]);
    const [prompt, setPrompt] = useState<any | null>(null)

    useEffect(() => {
        if (pass2Step === true) {
            const strPrompt = sessionStorage.getItem('prompt');
            //console.log("hehe")
            if (!strPrompt) { return; }
            const tmp = JSON.parse(strPrompt);
            setPrompt(tmp);
            console.log(tmp)
        }
    }, [pass2Step]);

    useEffect(() => {
        const readCache = async () => {
            const storedImageUrl = sessionStorage.getItem('uploadedImageUrl');
            console.log("step3 :", storedImageUrl)
            if (storedImageUrl) {
                try {
                    const response = await fetch(storedImageUrl);
                    const blob = await response.blob();
                    const fileName = "image.png";
                    const curfile = new File([blob], fileName, { type: blob.type });
                    setFile(curfile);
                } catch (err) {
                    console.error('Error fetching the image:', err);
                }
            }
        };
        if (prompt !== null) {
            readCache();
        }
    }, [prompt]);

    interface Real2AnimeResponse {
        images: string[]; // Array of base64-encoded images
    }

    const sendRequest = async () => {
        if (!file) {
            console.error('No file available to send');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('option_json', JSON.stringify(prompt));

        try {
            const response = await fetch('http://0.0.0.0:8000/real2anime/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                console.error('HTTP error:', response.status);
                setLoading(false);
                return;
            }

            const data: Real2AnimeResponse = await response.json(); // Type the response
            const imagesBase64 = data.images;

            const imageUrls = imagesBase64.map((base64String): string => {
                return `data:image/png;base64,${base64String}`;
            });

            // Display each image in the document
            imageUrls.forEach((url: string) => {
                const img = document.createElement('img');
                img.src = url;
            });

            console.log(imageUrls)

            setImages(imageUrls);
            setLoading(false);
        } catch (error) {
            console.error('Error:', error);
        }
    };

    useEffect(() => {
        if (file !== null) {
            sendRequest();
        }
    }, [file]);


    const [activeIndex, setActiveIndex] = useState<number>(2);
    const items: MenuItem[] = [
        { label: 'Step 1' },
        { label: 'Step 2' },
        { label: 'Step 3' },
    ];

    const router = useRouter();
    const HomeClick = async () => {
        // Clear all sessionStorage data
        sessionStorage.clear();
        router.push('/');
    };

    const handleDownload = (resultImg:string) => {
        const link = document.createElement('a');
        link.href = resultImg;
        link.download = 'custom-preview-image.jpg';
        link.click();
    };

    return (
        <div className="grid p-fluid input-demo">
            <div className="col-12">
                <div className="md:col-12">
                    <Steps model={items} activeIndex={activeIndex} onSelect={(e) => setActiveIndex(e.index)} />
                </div>

                <div className="card" style={{ display: loading ? 'block' : 'none' }}>
                    <h5>Waiting for processing...</h5>
                    <div className="card flex justify-content-center">
                        <ProgressSpinner />
                    </div>
                </div>
            </div>
            <div className='col-12 md:col-12' style={{ display: !loading ? 'block' : 'none' }}>
                <div className='card'>
                    <h5>Your result</h5>
                    <div className="border-1 surface-border border-round m-1 text-center py-5">
                        <div className="grid">
                            {images.map((url, index) => (
                                <div className='col-12 md:col-4' key={index} style={{ position: 'relative', marginBottom: '10px' }}>
                                    <Image
                                        src={url}
                                        width="100%"
                                        preview
                                    />
                                    <Button
                                        icon="pi pi-download"
                                        className="p-button-rounded p-button-secondary"
                                        onClick={() => handleDownload(url)}
                                        style={{ position: 'absolute', top: '10px', right: '10px' }}
                                        aria-label="Download"
                                    />
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
                <div className="flex justify-content-center gap-2">
                    <Tooltip target=".Finish" content="Back to main page" position="top" />
                    <Button
                        label="Finish"
                        className="Finish"
                        severity="success"
                        style={{ margin: '0.25em 0.25em', width: '150px' }}
                        onClick={HomeClick}
                    />
                </div>
            </div>
        </div>
    );
};

export default Step3;
