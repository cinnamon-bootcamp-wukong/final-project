'use client';

import React, { useRef, useState } from 'react';

import { Steps } from 'primereact/steps'
import { MenuItem } from 'primereact/menuitem';
import { Image } from 'primereact/image';
import { Button } from 'primereact/button';
import { Tooltip } from 'primereact/tooltip';
import { useRouter } from 'next/navigation';
import { Toast } from 'primereact/toast';  


const Step1 = () => {
    const toast = useRef<Toast>(null);

    const showSuccess = () => {
        toast.current?.show({severity:'success', summary: 'Success', detail:'Image uploaded', life: 3000});
    }
    const showError = () => {
        toast.current?.show({severity:'error', summary: 'Error', detail:'No file selected', life: 3000});
    }
    const router = useRouter();
    const [previewUrl, setPreviewUrl] = useState('/demo/images/galleria/upload.png');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);  // Allow state to be either File or null
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleUploadClick = () => {
        if (fileInputRef.current) {  // Check if fileInputRef.current is not null
            fileInputRef.current.click();  // Trigger the file input click
        }
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const fileUrl = URL.createObjectURL(file);
            setPreviewUrl(fileUrl);  // Set the preview URL to the uploaded file
            setSelectedFile(file);  // Store the selected file
            console.log("Selected file:", file);
            showSuccess();
        }
    };

    const handleNextClick = async () => {
        if (selectedFile) {
            // const formData = new FormData();
            // formData.append('file', selectedFile);

            // try {
            //     const response = await fetch('http://localhost:8000/file_upload/', {
            //         method: 'POST',
            //         body: formData
            //     });
            //     const result = await response.json();
            //     console.log('Server response:', result);
            // } catch (error) {
            //     console.error('Error uploading file:', error);
            // }
            router.push('step2');
        } else {
            console.log("No file selected");
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
        {
            label: 'Step 4'
        },
        {
            label: 'Step 5'
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
                            style={{ display: 'none' }}  // Hide the file input
                            accept="image/*"  // Only accept image files
                            onChange={handleFileChange}  // Handle file selection
                        />
                        <Tooltip target=".next" content="Next step" position="top" />
                        <Button 
                            label="Next"
                            className='next'
                            style={{ margin: '0.25em 0.25em', width: '150px' }}
                            onClick={handleNextClick}  // Attach the click handler for next
                        />
                    </div>

                </div>
            </div>
        </div>
    );
};

export default Step1;