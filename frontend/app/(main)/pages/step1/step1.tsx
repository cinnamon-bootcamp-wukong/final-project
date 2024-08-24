'use client';

import React, { useRef, useState, useEffect } from 'react';

import { Steps } from 'primereact/steps'
import { MenuItem } from 'primereact/menuitem';
import { Image } from 'primereact/image';
import { Button } from 'primereact/button';
import { Tooltip } from 'primereact/tooltip';
import { Toast } from 'primereact/toast';

interface step1Prob {
    setStep: React.Dispatch<React.SetStateAction<number>>;
}


const Step1: React.FC<step1Prob> = ({setStep}) => {
    const toast = useRef<Toast>(null);

    const showSuccess = () => {
        toast.current?.show({severity:'success', summary: 'Success', detail:'Image uploaded', life: 3000});
    }
    const showError = () => {
        toast.current?.show({severity:'error', summary: 'Error', detail:'No file selected', life: 3000});
    }
    const showErrorFace = () => {
        toast.current?.show({severity:'warn', summary: 'Try again', detail:'Cannot recognize your face', life: 3000});
    }

    const [previewUrl, setPreviewUrl] = useState('/demo/images/galleria/upload.png');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isFace, setIsFace] = useState<boolean>(false);
    const [tmpFile, setTmpFile] = useState<File | null>(null);

    const handleUploadClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        setIsFace(false);
        if (!file) {
            console.error('No file available to send');
            return;
        }
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch('http://0.0.0.0:8000/face_dect/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                console.error('HTTP error:', response.status);
                return;
            }

            const data = await response.json();
            if (data.is_face === true) {
                setIsFace(true)
            }
        } catch (error) {
            console.error('Error:', error);
        }

        setTmpFile(file)
    };

    useEffect(() => {
        if (isFace === true) {
            if (tmpFile === null) {
                console.error('No file available to send');
                return;
            }
            const fileUrl = URL.createObjectURL(tmpFile);
            setPreviewUrl(fileUrl);
            setSelectedFile(tmpFile);
            //sessionStorage.setItem('uploadedImageUrl', fileUrl);
        }
    }, [tmpFile]);

    useEffect(() => {
        if (isFace === true) {
            showSuccess()
        }
        else if (tmpFile !== null) {
            showErrorFace()
        }
    }, [tmpFile]);

    const handleNextClick = async () => {
        if (selectedFile) {
            const fileUrl = URL.createObjectURL(selectedFile)
            sessionStorage.setItem('uploadedImageUrl', fileUrl);
            console.log("step1 :", fileUrl)
            setStep(1);
        } else {
            showError()
        }
    };

    const [activeIndex, setActiveIndex] = useState<number>(0);
    const items: MenuItem[] = [
        {
            label: 'Step 1'
        },
        {
            label: 'Step 2'
        },
        {
            label: 'Step 3'
        },
    ];
    return (
        <div>
            <Toast ref={toast} />

            <div className="grid">
                <div className="md:col-12">
                        <Steps model={items} activeIndex={activeIndex} onSelect={(e) => setActiveIndex(e.index)} />
                </div>
                <div className="col-12">
                    <div className="card">
                        <h5>Upload your image</h5>

                        <div className="flex justify-content-center">
                            <Image src={previewUrl} width="280" />
                        </div>
                    </div>

                    <div className="flex justify-content-center gap-2">
                        <Tooltip target=".upload" content="Upload your face image" position="top" />
                        <Button
                            label="Upload"
                            className="upload"
                            style={{ margin: '0.25em 0.25em', width: '150px' }}
                            onClick={handleUploadClick}  // Attach the click handler
                        />
                        <input
                            type="file"
                            ref={fileInputRef}
                            style={{ display: 'none' }}
                            accept="image/*"
                            onChange={handleFileChange}
                        />
                        <Tooltip target=".next" content="Next step" position="top" />
                        <Button
                            label="Next"
                            className='next'
                            style={{ margin: '0.25em 0.25em', width: '150px' }}
                            onClick={handleNextClick}
                        />
                    </div>

                </div>
            </div>
        </div>
    );
};

export default Step1;